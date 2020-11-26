"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class xsim_surrounding_info(object):
    __slots__ = ["ntime", "num_of_surrounding_vehicle", "id", "x_pos_rel", "y_pos_rel", "x_vel_rel", "y_vel_rel", "length", "width", "heading"]

    def __init__(self):
        self.ntime = 0
        self.num_of_surrounding_vehicle = 0
        self.id = []
        self.x_pos_rel = []
        self.y_pos_rel = []
        self.x_vel_rel = []
        self.y_vel_rel = []
        self.length = []
        self.width = []
        self.heading = []

    def encode(self):
        buf = BytesIO()
        buf.write(xsim_surrounding_info._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qi", self.ntime, self.num_of_surrounding_vehicle))
        buf.write(struct.pack('>%di' % self.num_of_surrounding_vehicle, *self.id[:self.num_of_surrounding_vehicle]))
        buf.write(struct.pack('>%df' % self.num_of_surrounding_vehicle, *self.x_pos_rel[:self.num_of_surrounding_vehicle]))
        buf.write(struct.pack('>%df' % self.num_of_surrounding_vehicle, *self.y_pos_rel[:self.num_of_surrounding_vehicle]))
        buf.write(struct.pack('>%df' % self.num_of_surrounding_vehicle, *self.x_vel_rel[:self.num_of_surrounding_vehicle]))
        buf.write(struct.pack('>%df' % self.num_of_surrounding_vehicle, *self.y_vel_rel[:self.num_of_surrounding_vehicle]))
        buf.write(struct.pack('>%df' % self.num_of_surrounding_vehicle, *self.length[:self.num_of_surrounding_vehicle]))
        buf.write(struct.pack('>%df' % self.num_of_surrounding_vehicle, *self.width[:self.num_of_surrounding_vehicle]))
        buf.write(struct.pack('>%df' % self.num_of_surrounding_vehicle, *self.heading[:self.num_of_surrounding_vehicle]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != xsim_surrounding_info._get_packed_fingerprint():
            raise ValueError("Decode error")
        return xsim_surrounding_info._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = xsim_surrounding_info()
        self.ntime, self.num_of_surrounding_vehicle = struct.unpack(">qi", buf.read(12))
        self.id = struct.unpack('>%di' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        self.x_pos_rel = struct.unpack('>%df' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        self.y_pos_rel = struct.unpack('>%df' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        self.x_vel_rel = struct.unpack('>%df' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        self.y_vel_rel = struct.unpack('>%df' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        self.length = struct.unpack('>%df' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        self.width = struct.unpack('>%df' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        self.heading = struct.unpack('>%df' % self.num_of_surrounding_vehicle, buf.read(self.num_of_surrounding_vehicle * 4))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if xsim_surrounding_info in parents: return 0
        tmphash = (0x3e290c2b60d0415c) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if xsim_surrounding_info._packed_fingerprint is None:
            xsim_surrounding_info._packed_fingerprint = struct.pack(">Q", xsim_surrounding_info._get_hash_recursive([]))
        return xsim_surrounding_info._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

