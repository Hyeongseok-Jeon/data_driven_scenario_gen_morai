"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class xsim_lane_change_status(object):
    __slots__ = ["ntime", "num_of_vehicle", "id", "status", "dir"]

    def __init__(self):
        self.ntime = 0
        self.num_of_vehicle = 0
        self.id = []
        self.status = []
        self.dir = []

    def encode(self):
        buf = BytesIO()
        buf.write(xsim_lane_change_status._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qi", self.ntime, self.num_of_vehicle))
        buf.write(struct.pack('>%di' % self.num_of_vehicle, *self.id[:self.num_of_vehicle]))
        buf.write(struct.pack('>%db' % self.num_of_vehicle, *self.status[:self.num_of_vehicle]))
        buf.write(struct.pack('>%db' % self.num_of_vehicle, *self.dir[:self.num_of_vehicle]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != xsim_lane_change_status._get_packed_fingerprint():
            raise ValueError("Decode error")
        return xsim_lane_change_status._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = xsim_lane_change_status()
        self.ntime, self.num_of_vehicle = struct.unpack(">qi", buf.read(12))
        self.id = struct.unpack('>%di' % self.num_of_vehicle, buf.read(self.num_of_vehicle * 4))
        self.status = struct.unpack('>%db' % self.num_of_vehicle, buf.read(self.num_of_vehicle))
        self.dir = struct.unpack('>%db' % self.num_of_vehicle, buf.read(self.num_of_vehicle))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if xsim_lane_change_status in parents: return 0
        tmphash = (0x259d4f6ce2b23708) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if xsim_lane_change_status._packed_fingerprint is None:
            xsim_lane_change_status._packed_fingerprint = struct.pack(">Q", xsim_lane_change_status._get_hash_recursive([]))
        return xsim_lane_change_status._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)
