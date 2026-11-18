#!/usr/bin/env python3
import copy
import math
import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag, StrEnum
from itertools import zip_longest
from typing import assert_never, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

import counter_priors

WINDOW_SIZES: tuple[tuple[int, int], ...] = (
    (1920, 1080), (2560, 1440), (1366, 768), (1600, 900), (1680, 1050),
    (1024, 768), (1440, 900), (1280, 800), (2560, 1080), (1280, 1024),
    (1920, 1440), (1360, 768), (1152, 864), (800, 600), (1280, 960),
    (1920, 1200), (3840, 2160), (3440, 1440), (1400, 1050), (1600, 1200),
    (2560, 1600), (1600, 1024), (1280, 720), (1536, 864)
)

POLLING_RATES: tuple[int, ...] = (
    125, 250, 500, 1000, 2000, 4000, 8000
)

TIME_GRID_OPTS: tuple[float, ...] = (
    0.0, 50/3, 25.0, 25/2, 100/3, 20.0
)

PLAY_RATES: tuple[float, ...] = (
    1.0, 1.5, 0.75
)

def clamp[T: SupportsRichComparison](x: T, _min: T, _max: T) -> T:
    return max(_min, min(_max, x))

class BitReader:
    def __init__(self, data: bytes) -> None:
        self.n = int.from_bytes(data, "little")
        self.offset = 0

    def read_bits(self, width: int) -> int:
        val = (self.n >> self.offset) & ((1 << width) - 1)
        self.offset += width
        return val

    def read_bit(self) -> int:
        return self.read_bits(1)

    def read_f32(self) -> float:
        return struct.unpack(">f", self.read_bits(32).to_bytes(4))[0]

    def read_bucket(self, *bucket_sizes: int):
        ret = 0
        for bsize in bucket_sizes:
            val = self.read_bits(bsize)
            ret += val
            if val < 2 ** bsize - 1:
                return ret

        e = self.read_bits(5)
        r = self.read_bits(e)
        return ret + 2 ** e + r - 1

    def read_length_bucket(
        self,
        payload_bit_length: int,
        start_bit: int,
    ) -> int:
        code = self.read_bits(payload_bit_length)
        if code == 2 ** payload_bit_length - 1:
            r = self.read_bits(5)
            if r < start_bit:
                length = r
            else:
                length = r + code
        else:
            length = start_bit + code

        if length == 0:
            return 0

        payload = self.read_bits(length - 1)
        return 2 ** (length - 1) + payload

    def read_byte(self) -> int:
        return self.read_bits(8)

    def read_small(self, n):
        v = self.read_bits(n)
        if v < 2 ** n - 1:
            return v

        length = n - 1
        while self.read_bits(1) != 0:
            length += 1

        if length == n - 1:
            return 2 ** n - 1

        return 2 ** length + self.read_bits(length)

    def skip(self, n: int) -> None:
        self.offset += n

    def skip_to_next_byte_boundary(self) -> None:
        off_boundary = self.offset & 7
        if off_boundary:
            self.skip(8 - off_boundary)

    def preview_bits(self, count: int) -> list[int]:
        vals = []
        for width in range(count):
            val =(self.n >> self.offset + width) & 1
            vals.append(val)
        return vals

def zigzag(v: int) -> int:
    if v > 0:
        return 2 * v
    else:
        return -2 * v - 1

@dataclass
class TimeGrid:
    period: float

class ContainerTag(IntEnum):
    FRAMES = 1
    OSR_META_DATA = 2
    OSR_VERBATIM = 3

@dataclass
class Container:
    tag: ContainerTag
    size: int

@dataclass
class CustomContainer:
    name: str
    size: int

@dataclass
class RationalParams:
    D: int
    a: int
    b: int

@dataclass
class MouseParams:
    sensitivity: float
    polling_rate: float
    play_rate: float
    unrounded_subtraction: bool
    window: tuple[float, float]

class TabletMapping(IntEnum):
    DESKTOP = 0
    WINDOW = 1

@dataclass
class AbsoluteParams:
    # TODO: rumoi said maybe some of this stuff should
    #       not be only on abs params
    tablet_mapping: TabletMapping
    x87: bool
    unrounded_subtraction: bool
    rounding_variant: bool
    span: float
    origin: int
    window_half_size: int
    area_speed: float
    window: tuple[float, float]


class DeviceKind(IntEnum):
    RATIONAL = 0
    MOUSE = 1
    ABSOLUTE = 2
    TEXT = 3

@dataclass(kw_only=True)
class Device:
    rational: RationalParams | None = None
    mouse: MouseParams | None = None
    absolute: AbsoluteParams | None = None
    # TEXT is defined as the lack of any params

    @property
    def kind(self) -> DeviceKind:
        if self.rational:
            return DeviceKind.RATIONAL
        if self.mouse:
            return DeviceKind.MOUSE
        if self.absolute:
            return DeviceKind.ABSOLUTE
        return DeviceKind.TEXT

    def is_cloneable(self) -> bool:
        return bool(
            self.rational
            or (
                self.mouse
                and self.mouse.window in WINDOW_SIZES
            )
            and not self.absolute
        )

@dataclass
class Segment:
    length: int | None
    device: Device

class AxisDirection(StrEnum):
    X = "X"
    Y = "Y"

@dataclass
class Axis:
    direction: AxisDirection
    segments: list[Segment]


RATE = tuple[int, ...](131_072 // (2 * n + 3) for n in range(32))
COUNTER_LIMIT = 31

@dataclass
class Counter:
    probability: int = 1024
    count: int = 0

    def update(self, y: int) -> None:
        self.probability = self.probability + math.floor(
            ((2047 * y - self.probability) * RATE[self.count]) // 2 ** 16
        )
        if self.count < COUNTER_LIMIT:
            self.count += 1


def context_hash(input_id: int, context: int) -> int:
    h = (input_id + 1) * 0x9e3779b97F4a7c15 ^ context * 0xd6e8feb86659fd93
    h &= 0xffffffffffffffff
    h ^= h >> 29
    h *= 0xbf58476d1ce4e5b9
    h &= 0xffffffffffffffff
    h ^= h >> 32
    return h

@dataclass
class LineTable:
    counters: list[Counter]
    width: int
    mask: int

    def __init__(self, bits: int, width: int) -> None:
        self.width = width
        self.counters = [Counter() for _ in range(width << bits)]
        self.mask = (1 << bits) - 1

    def line(self, h: int) -> list[Counter]:
        i = h & self.mask
        return self.counters[i * self.width:]

START_WEIGHT = 19661

@dataclass
class Model:
    table: LineTable
    weights: list[int]

    def __init__(
        self,
        num_inputs: int,
        table_bits: int,
        width: int,
        mix_sets: int,
    ) -> None:
        self.table = LineTable(table_bits, width)
        mixed = num_inputs + 1
        self.weights = [START_WEIGHT for _ in range(mix_sets * mixed)]

@dataclass
class Models:
    keys_changed: Model

class EffectiveMode(IntEnum):
    OSU = 0
    TAIKO = 1
    FRUITS = 2
    MANIA = 3
    OSU_RELAX = 4
    TAIKO_RELAX = 5
    FRUITS_RELAX = 6
    OSU_AUTOPILOT = 7
    E_MODE_MAX = 8

class EncodeInfoBits(IntFlag):
    E_MODE_BITS = 15

    HAS_OSR_HEADER = 1 << 4

    TRAILING_COMMA = 1 << 5

    COMPRESSION_PRIORITY_MASK = 3 << 6
    COMPRESSION_PRIORITY_SIZE = 0 << 6
    COMPRESSION_PRIOTITY_BALANCED = 1 << 6
    COMPRESSION_PRIORITY_SPEED = 2 << 6

    REPLAY_VERBATIM = 1 << 8
    TEXT_SHORTEST = 1 << 9

class CompressionPriority(IntEnum):
    SIZE = 0
    BALANCED = 1
    SPEED = 2

@dataclass
class EncodeInfo:
    e_mode: EffectiveMode
    has_osr_header: bool
    trailing_comma: bool
    priority: CompressionPriority
    replay_verbatim: bool
    text_shortest: bool

def parse_encode_info(raw: int) -> EncodeInfo:
    return EncodeInfo(
        e_mode=EffectiveMode(raw & EncodeInfoBits.E_MODE_BITS),
        has_osr_header=raw & EncodeInfoBits.HAS_OSR_HEADER == 1,
        trailing_comma=raw & EncodeInfoBits.TRAILING_COMMA == 1,
        priority=CompressionPriority(raw & EncodeInfoBits.COMPRESSION_PRIORITY_MASK),
        replay_verbatim=raw & EncodeInfoBits.REPLAY_VERBATIM == 1,
        text_shortest=raw & EncodeInfoBits.TEXT_SHORTEST == 1,
    )

def is_relax(e_mode: EffectiveMode) -> bool:
    return e_mode in {
        EffectiveMode.OSU_RELAX,
        EffectiveMode.TAIKO_RELAX,
        EffectiveMode.FRUITS_RELAX
    }

class PriorSet(IntEnum):
    SET_RATIONAL = 0
    SET_MOUSE = 1
    SET_ABSOLUTE = 2
    SET_RELAX_RATIONAL = 3
    SET_RELAX_MOUSE = 4
    SET_RELAX_ABSOLUTE = 5
    SET_RELAX_OTHER = 6
    SET_NO_DEVICE = 7
    SET_TAIKO = 8


def get_prior_set(axes: dict[AxisDirection, Axis], encode_info: EncodeInfo) -> PriorSet:
    device_kind: DeviceKind | None = None
    for axis in axes.values():
        found = axis.segments[0].device.kind
        device_kind = max(device_kind, found) if device_kind else found

    if device_kind is None or device_kind is DeviceKind.TEXT:
        return (
            PriorSet.SET_RELAX_OTHER
            if is_relax(encode_info.e_mode)
            else PriorSet.SET_NO_DEVICE
        )

    return PriorSet(
        (
            PriorSet.SET_RELAX_RATIONAL
            if is_relax(encode_info.e_mode)
            else PriorSet.SET_RATIONAL
        ) + device_kind
    )

@dataclass
class Prior:
    start: list[int]
    keys: list[int]
    values: list[int]
    weights_start: list[int]
    weights: list[int]

def seed_priors(
    models: list[Model],
    axes: dict[AxisDirection, Axis],
    encode_info: EncodeInfo,
) -> None:
    prior_set = get_prior_set(axes, encode_info)
    prior = Prior(
        start=counter_priors.FRAMES_PRIOR_START[prior_set],
        keys=counter_priors.FRAMES_PRIOR_KEYS,
        values=counter_priors.FRAMES_PRIOR_VALUES,
        weights_start=counter_priors.FRAMES_PRIOR_WEIGHT_START[prior_set],
        weights=counter_priors.FRAMES_PRIOR_WEIGHTS,
    )

    for m in range(1): # counter_priors.FRAMES_PRIOR_MODELS):
        model = models[m]
        if not model.table.counters:
            continue

        # counters
        for j in range(prior.start[m], prior.start[m + 1]):
            entry = prior.values[j]
            slot = entry >> 24
            count = min((entry >> 16) & 0xff, COUNTER_LIMIT)
            probability = (entry & 0xffff) >> 5
            new_counter = Counter(probability, count)
            model.table.line(prior.keys[j])[slot] = new_counter

        # TODO: counters_only for balanced mode

        # weights
        w0, w1 = prior.weights_start[m], prior.weights_start[m + 1]
        if len(model.weights) == w1 - w0:
            model.weights = prior.weights[w0:w1].copy()


SQUASH_TABLE: tuple[int, ...] = (
    1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546,
    2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068,
    4079, 4085, 4089, 4092, 4093, 4094,
)

def squash(d: int) -> int:
    if d < -2047:
        return 1
    if d > 2047:
        return 4095
    i = math.floor(d / 128) + 16
    r = d - 128 * math.floor(d / 128)
    return (SQUASH_TABLE[i] * (128 - r) + SQUASH_TABLE[i + 1] * r + 64) // 128

# p = (a(128 - r) + b * r + 64) / 128
# 128p = a(128 - r) + b * r + 64
# 128p - 64 = a(128 - r) + (b * r)
# 128p - 64 = a128 - (a * r) + (b * r)
# 128p - 64 - a128 = -(a * r) + (b * r)
# 128p - 64 - a128 = (-a + b) * r
# 128p - 64 - a128 = (b - a) * r
# (128p - 64 - a128) / (b - a) = r

def stretch(p: int) -> int:
    i = 0
    while i < len(SQUASH_TABLE) - 2 and SQUASH_TABLE[i + 1] < p:
        i += 1
    a, b = SQUASH_TABLE[i], SQUASH_TABLE[i + 1]
    r = math.ceil((128 * p - 64 - 128 * a) / (b - a))
    d = 128 * (i - 16) + r
    return clamp(d, -2047, 2047)

def mix(counters: list[Counter], weights: list[int]) -> int:
    tot = 0
    for counter, weight in zip(counters, weights):
        pi = counter.probability * 32
        xi = stretch(math.floor(pi / 16))
        tot += xi * weight
    return squash(clamp(math.floor(tot // 2 ** 16), -2047, 2047))

class BinaryCoder:
    def __init__(self, reader: BitReader) -> None:
        self.x1, self.x2, self.x = 0, 0xffffffff, 0
        self.reader = reader

    def code_bit(self, p: int) -> int:
        xmid = self.x1 + math.floor(
            (self.x2 - self.x1) * p / 2 ** 16
        )

        y = self.reader.read_bit()
        if y == 1:
            self.x2 = xmid
        else:
            self.x1 = xmid + 1

        while (self.x1 ^ self.x2) & 0xff000000 == 0:
            self.x1 = (self.x1 << 8) & 0xffffffff
            self.x2 = (self.x2 << 8) & 0xffffffff | 255
            self.x = (self.x << 8) & 0xffffffff | self.reader.read_byte()

        return y


@dataclass
class NativeGrid:
    step: int
    offset: int


def main() -> int:
    with open("example.orz", "rb") as f:
        orz_data = f.read()
    reader = BitReader(orz_data)
    magic = reader.read_byte()
    assert magic == 0xE6
    version = reader.read_bucket(4, 8)
    crc = reader.read_bits(32)
    encode_info = parse_encode_info(reader.read_length_bucket(4, 0))
    seeded = reader.read_bit()
    if seeded:
        seed = reader.read_length_bucket(3, 18)
    frame_count = reader.read_bucket(16)

    # containers
    containers: list[Container] = []
    previous_tag = 0
    while reader.read_bit() == 1:
        tag = previous_tag + 1
        if reader.read_bit() == 1:
            tag += reader.read_bucket(4)
        previous_tag = tag
        if tag > 255:
            raise ValueError("Tag > 255 is invalid by orz spec")
        size = reader.read_bucket(16)
        if tag not in ContainerTag:
            continue

        container = Container(
            tag=ContainerTag(tag),
            size=size,
        )
        containers.append(container)

    # custom containers
    custom_containers: list[CustomContainer] = []
    if reader.read_bit() == 1:
        custom_container_count = reader.read_bucket(4) + 1
        for _ in range(custom_container_count):
            name = "".join(
                chr(reader.read_bits(7))
                for _ in range(reader.read_bits(5))
            )
            size = reader.read_bucket(16)
            custom_container = CustomContainer(name, size)
            custom_containers.append(custom_container)

    # TODO: assertions
    # TODO: CRC

    # pad to byte boundary
    reader.skip_to_next_byte_boundary()

    # parse containers

    for container in containers:
        # TODO: for each container, consider `size`
        if container.tag is ContainerTag.FRAMES:

            # delta table
            deltas = []
            n = reader.read_small(5)
            for _ in range(min(2, n)):
                bit = reader.read_bit()
                if bit == 1:
                    delta = reader.read_bits(5)
                else:
                    delta = reader.read_length_bucket(4, 0)
                deltas.append(delta)

            s = reader.read_bits(5)
            w = reader.read_length_bucket(3, 0)
            deltas += [i for i in range(s) if i not in deltas]
            for v in range(s, w):
                if reader.read_bits(1) and len(deltas) < n:
                    deltas.append(v)

            while len(deltas) != n:
                sign = reader.read_bit()
                magnitude = reader.read_length_bucket(4, 0)
                delta = -magnitude if sign == 1 else +magnitude
                deltas.append(delta)

            # time grid

            period_choice = reader.read_bits(3)
            if period_choice < 6:
                period = TIME_GRID_OPTS[period_choice]
            else:
                num = reader.read_bucket(7) + 1
                den = reader.read_bucket(3) + 1
                period = num / den

            time_grid = TimeGrid(period)

            # device models

            segment_count = 1
            if reader.read_bit() == 1:
                segment_count += reader.read_bits(5) + 1

            seg_lengths: list[int] = [
                reader.read_length_bucket(4, 0) + 1
                for _ in range(segment_count - 1) # TODO
            ]

            axes: dict[AxisDirection, Axis] = {
                direction: Axis(direction=direction, segments=[])
                for direction in AxisDirection
            }
            for direction, axis in axes.items():
                for seg_num, seg_length in zip_longest(
                    range(segment_count),
                    seg_lengths,
                ):

                    if direction == AxisDirection.Y:
                        # If clone bit is present, copy x
                        x_axis = axes[AxisDirection.X]
                        x_device = x_axis.segments[seg_num].device
                        if (
                            x_device.is_cloneable()
                            and reader.read_bit() == 1
                        ):
                            segment = Segment(
                                length=seg_length,
                                device=copy.copy(x_device),
                            )
                            axis.segments.append(segment)
                            continue

                    mode = DeviceKind(reader.read_bits(2))

                    if mode is DeviceKind.RATIONAL:
                        D = reader.read_bucket(4, 8, 16)
                        a = reader.read_bucket(4, 8, 16)
                        b = zigzag(reader.read_bucket(4, 8, 16))
                        device = Device(
                            rational=RationalParams(D, a, b),
                        )

                    elif mode is DeviceKind.MOUSE:
                        if reader.read_bit() == 1:
                            sensitivity = reader.read_f32()
                        else:
                            hundredths = reader.read_bucket(4, 8, 16)
                            sensitivity = hundredths / 100.0

                        poll_choice = reader.read_bits(3)
                        if poll_choice == 7:
                            polling_rate = reader.read_bits(13)
                        else:
                            polling_rate = POLLING_RATES[poll_choice]

                        play_rate_opt = reader.read_bits(2)
                        if play_rate_opt == 3:
                            play_rate = struct.unpack(">f", reader.read_bits(31).to_bytes(4))[0]
                            if (
                                play_rate == 0
                                or math.isnan(play_rate)
                                or math.isinf(play_rate)
                            ):
                                raise ValueError(f"Invalid play rate {play_rate}")
                        else:
                            play_rate = PLAY_RATES[play_rate_opt]

                        unrounded_subtraction = reader.read_bit() == 1

                        if reader.read_bit() == 1:
                            window_choice = reader.read_bits(5)
                            window = WINDOW_SIZES[window_choice]
                        else:
                            window_w = reader.read_f32()
                            window_h = reader.read_f32()
                            window = (window_w, window_h)

                        device = Device(
                            mouse=MouseParams(
                                sensitivity=sensitivity,
                                polling_rate=polling_rate,
                                play_rate=play_rate,
                                unrounded_subtraction=unrounded_subtraction,
                                window=window,
                            ),
                        )

                    elif mode is DeviceKind.ABSOLUTE:
                        mapping = TabletMapping(reader.read_bit())
                        x87 = reader.read_bit() == 1
                        unrounded_subtraction = reader.read_bit() == 1
                        rounding_variant = reader.read_bit() == 1
                        span = reader.read_bucket(4, 8, 16)
                        origin = zigzag(reader.read_bucket(4, 8, 16))
                        window_half_size = reader.read_bucket(4, 8, 16)
                        if reader.read_bit() == 1:
                            hundredths = reader.read_bucket(4, 8, 16)
                            area_speed = hundredths / 100.0
                        else:
                            area_speed = reader.read_f32()
                        if reader.read_bit() == 1:
                            window_choice = reader.read_bits(5)
                            window = WINDOW_SIZES[window_choice]
                        else:
                            window_w = reader.read_f32()
                            window_h = reader.read_f32()
                            window = (window_w, window_h)

                        device = Device(
                            absolute=AbsoluteParams(
                                tablet_mapping=mapping,
                                x87=x87,
                                unrounded_subtraction=unrounded_subtraction,
                                rounding_variant=rounding_variant,
                                span=span,
                                origin=origin,
                                window_half_size=window_half_size,
                                area_speed=area_speed,
                                window=window,
                            ),
                        )

                    elif mode is DeviceKind.TEXT:
                        # There are no parameters for text
                        device = Device()
                        pass

                    else:
                        assert_never(mode)

                    segment = Segment(seg_length, device)
                    axis.segments.append(segment)

            # TODO: ap read

            # native grid

            native_grid = []
            for axis in range(2):
                if reader.read_bit() == 1:
                    step = reader.read_bits(48)
                    offset = reader.read_bits(64)
                    grid = NativeGrid(step, offset)
                    native_grid.append(grid)

            # std default pos

            std_default_pos = reader.read_bit() == 1
            std_default_time = (
                reader.read_bit() == 1
                if std_default_pos == 1
                else None
            )

            reader.skip_to_next_byte_boundary()

            # coded body

            x_escapes = reader.read_bit() == 1
            y_escapes = reader.read_bit() == 1

            # keys changed
            b = min(frame_count, 16384).bit_length()
            keys_changed = Model(
                num_inputs=5,
                table_bits=clamp(b+4, 16, 22),
                width=1,
                mix_sets=64,
            )
            # models = Models(keys_changed)
            seed_priors([keys_changed], axes, encode_info)

            coder = BinaryCoder(reader)

            for frame in range(frame_count):
                p_m = mix(counters, weights)
                p = clamp(16 * p_m, 32, 65504)
                y = coder.code_bit(p)
                # TODO: update the counters and weights

        elif container.tag is ContainerTag.OSR_META_DATA:
            reader.skip(container.size)

        elif container.tag is ContainerTag.OSR_VERBATIM:
            reader.skip(container.size)

        else:
            assert_never(container.tag)

    return 0

if __name__ == "__main__":
    exit(main())
