import time
import wave
import select
import socket
import audioop
import logging
import threading
import traceback

from collections import defaultdict

from . import rtp
from .rtp import SilencePacket
from .opus import Decoder as OpusDecoder, OpusRouter

try:
    import nacl.secret
    from nacl.exceptions import CryptoError
except ImportError:
    pass

log = logging.getLogger(__name__)

__all__ = [
    'AudioSink',
    'WaveSink',
    'PCMVolumeTransformerFilter',
    'ConditionalFilter',
    'TimedFilter',
    'UserFilter',
]

class AudioSink:
    def __del__(self):
        self.cleanup()

    def write(self, data):
        raise NotImplementedError

    def wants_opus(self):
        return False

    def cleanup(self):
        pass

class WaveSink(AudioSink):
    def __init__(self, destination):
        self._file = wave.open(destination, 'wb')
        self._file.setnchannels(OpusDecoder.CHANNELS)
        self._file.setsampwidth(OpusDecoder.SAMPLE_SIZE//OpusDecoder.CHANNELS)
        self._file.setframerate(OpusDecoder.SAMPLING_RATE)

    def write(self, data):
        self._file.writeframes(data.data)

    def cleanup(self):
        try:
            self._file.close()
        except:
            pass

class PCMVolumeTransformerFilter(AudioSink):
    def __init__(self, destination, volume=1.0):
        if not isinstance(destination, AudioSink):
            raise TypeError('expected AudioSink not {0.__class__.__name__}.'.format(destination))

        if destination.wants_opus():
            raise ClientException('AudioSink must not request Opus encoding.')

        self.destination = destination
        self.volume = volume

    @property
    def volume(self):
        """Retrieves or sets the volume as a floating point percentage (e.g. 1.0 for 100%)."""
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = max(value, 0.0)

    def write(self, data):
        data = audioop.mul(data.data, 2, min(self._volume, 2.0))
        self.destination.write(data)

# I need some sort of filter sink with a predicate or something
# Which means I need to sort out the write() signature issue
# Also need something to indicate a sink is "done", probably
# something like raising an exception and handling that in the write loop
# Maybe should rename some of these to Filter instead of Sink

class ConditionalFilter(AudioSink):
    def __init__(self, destination, predicate):
        self.destination = destination
        self._predicate = predicate

    def write(self, data):
        if self._predicate(data):
            self.destination.write(data)

class TimedFilter(ConditionalFilter):
    def __init__(self, destination, duration):
        super().__init__(destination, self._predicate)
        self.duration = duration

    def _predicate(self, data):
        return # TODO: return elapsed < duration

class UserFilter(ConditionalFilter):
    def __init__(self, destination, user):
        super().__init__(destination, self._predicate)
        self.user = user

    def _predicate(self, data):
        return data.user == self.user

# rename 'data' to 'payload'? or 'opus'? something else?
class VoiceData:
    __slots__ = ('data', 'user', 'packet')

    def __init__(self, data, user, packet):
        self.data = data
        self.user = user
        self.packet = packet

class AudioReader(threading.Thread):
    def __init__(self, sink, client):
        threading.Thread.__init__(self)
        self.daemon = True
        self._sink = sink
        self.client = client

        self.box = nacl.secret.SecretBox(bytes(client.secret_key))
        self._decrypt_rtp = getattr(self, '_decrypt_rtp_' + client.mode)
        self._decrypt_rtcp = getattr(self, '_decrypt_rtcp_' + client.mode)

        self._connected = client._connected
        self._current_error = None
        self._decoders = defaultdict(lambda: OpusRouter(self._write_to_sink))

        self._end = threading.Event()
        self._decoder_lock = threading.Lock()

        self.packets = 0
        self.fails = 0

    def stop(self, *, wait=False):
        self._end.set()
        if wait:
            self.join()

    def _decrypt_rtp_xsalsa20_poly1305(self, packet):
        nonce = bytearray(24)
        nonce[:12] = packet.header
        result = self.box.decrypt(bytes(packet.data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305(self, data):
        nonce = bytearray(24)
        nonce[:8] = data[:8]
        result = self.box.decrypt(data[8:], bytes(nonce))

        return data[:8] + result

    def _decrypt_rtp_xsalsa20_poly1305_suffix(self, packet):
        nonce = packet.data[-24:]
        voice_data = packet.data[:-24]
        result = self.box.decrypt(bytes(voice_data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305_suffix(self, data):
        nonce = data[-24:]
        header = data[:8]
        result = self.box.decrypt(data[8:-24], nonce)

        return header + result

    def _reset_decoder(self, ssrc):
        with self._decoder_lock:
            if ssrc is None:
                for decoder in self._decoders.values():
                    decoder.reset()
            else:
                d = self._decoders.get(ssrc)
                if d:
                    d.reset()

    def _ssrc_removed(self, ssrc):
        # An user has disconnected but there still may be
        # packets from them left in the buffer to read
        # For now we're just going to kill the decoder and see how that works out
        # I *think* this is the correct way to do this
        # Depending on how many leftovers I end up with I may reconsider

        with self._decoder_lock:
            print(f"Removing decoder for ssrc {ssrc}")
            print(self._decoders.keys())
            decoder = self._decoders.pop(ssrc, None)

            if decoder is None:
                print(f"!!! No decoder for ssrc {ssrc} was found?")
            else:
                print(f"Removed decoder {ssrc}")
                if decoder.rtpheap:
                    print(f"Decoder had {len(decoder.rtpheap)} packets remaining")

    def _get_user(self, packet):
        return self.client._ssrcs.get(packet.ssrc)

    def _write_to_sink(self, pcm, opus, packet):
        try:
            data = opus if self._sink.wants_opus() else pcm
            user = self._get_user(packet)
            self._sink.write(VoiceData(data, user, packet))
        except:
            traceback.print_exc()
            # insert optional error handling here

    def _notify_decoders(self):
        with self._decoder_lock:
            # potential optimization here, only notify those in current channel
            for ssrc, decoder in self._decoders.items():
                if ssrc not in self.client._ssrcs:
                    # buffer unknown packets for now, needs testing
                    # also need to make sure to remove decoders when ssrcs die
                    continue
                try:
                    decoder.notify()
                except Exception as e:
                    self.fails += 1
                    traceback.print_exc()
                    # print(f"packets: {self.packets}, fails: {self.fails}")

    def _do_run(self):
        print("Starting socket loop")
        while not self._end.is_set():
            if not self._connected.is_set():
                self._connected.wait()

            ready, _, err = select.select([self.client.socket], [],
                                          [self.client.socket], 0.1)
            if not ready:
                if err:
                    print("Socket error")
                self._notify_decoders()
                continue

            try:
                raw_data = self.client.socket.recv(4096)
            except socket.error as e:
                t0 = time.time()
                # TODO: Catch and ignore "not a socket" error
                print(f"Socket error in reader thread: {e} {t0}")

                with self.client._connecting:
                    timed_out = self.client._connecting.wait(20)

                if not timed_out:
                    raise
                elif self.client.is_connected():
                    print(f"Reconnected in {time.time()-t0:.4f}s")
                    continue
                else:
                    raise

            try:
                packet = None

                if not rtp.is_rtcp(raw_data):
                    packet = rtp.decode(raw_data)
                    packet.decrypted_data = self._decrypt_rtp(packet)
                else:
                    packet = rtp.decode(self._decrypt_rtcp(raw_data))
                    if not isinstance(packet, rtp.ReceiverReportPacket):
                        print(packet)
                    continue # oh right I don't have any way to send these yet

            except CryptoError:
                log.exception("CryptoError decoding packet %s", packet)
                continue

            except:
                # Dropped/invalid (idk if packets can be non-rtp)
                log.exception("Unknown error decoding packet")
                traceback.print_exc()

            else:
                # Do I hold these packets or drop them?
                if packet.ssrc not in self.client._ssrcs:
                    log.debug("Unknown user for ssrc %s", packet.ssrc)

                    # TODO
                    # As expected, this is racy with users disconnecting
                    # I'm not sure what to do about this yet
                    # The problem is this is a 2-part problem
                    # I can capture packets for processing before I get
                    # an ssrc-userid mapping and thats fine, but it also
                    # recreates the decoder when there are leftover packets
                    # after someone disconnects.

                self.packets += 1
                self._decoders[packet.ssrc].feed(packet)

            finally:
                # print(self._decoders.items())
                self._notify_decoders()

        # flush decoders?

    def run(self):
        try:
            self._do_run()
        except socket.error as e:
            self.stop()
        except Exception as e:
            traceback.print_exc()
            self._current_error = e
            self.stop()
        finally:
            try:
                self._sink.cleanup()
            except:
                # pass
                # XXX: Testing only
                traceback.print_exc()
