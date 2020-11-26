"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class xsim_ego_info(object):
    __slots__ = ["ntime", "x_pos_ego", "y_pos_ego", "speed_ego", "heading_ego", "blinker_info", "steering_angle", "fl_wheel_vel", "fr_wheel_vel", "rl_wheel_vel", "rr_wheel_vel"]

    def __init__(self):
        self.ntime = 0
        self.x_pos_ego = 0.0
        self.y_pos_ego = 0.0
        self.speed_ego = 0.0
        self.heading_ego = 0.0
        self.blinker_info = 0
        self.steering_angle = 0.0
        self.fl_wheel_vel = 0.0
        self.fr_wheel_vel = 0.0
        self.rl_wheel_vel = 0.0
        self.rr_wheel_vel = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(xsim_ego_info._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qffffifffff", self.ntime, self.x_pos_ego, self.y_pos_ego, self.speed_ego, self.heading_ego, self.blinker_info, self.steering_angle, self.fl_wheel_vel, self.fr_wheel_vel, self.rl_wheel_vel, self.rr_wheel_vel))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != xsim_ego_info._get_packed_fingerprint():
            raise ValueError("Decode error")
        return xsim_ego_info._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = xsim_ego_info()
        self.ntime, self.x_pos_ego, self.y_pos_ego, self.speed_ego, self.heading_ego, self.blinker_info, self.steering_angle, self.fl_wheel_vel, self.fr_wheel_vel, self.rl_wheel_vel, self.rr_wheel_vel = struct.unpack(">qffffifffff", buf.read(48))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if xsim_ego_info in parents: return 0
        tmphash = (0x7ebb28ae466b0de) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if xsim_ego_info._packed_fingerprint is None:
            xsim_ego_info._packed_fingerprint = struct.pack(">Q", xsim_ego_info._get_hash_recursive([]))
        return xsim_ego_info._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

