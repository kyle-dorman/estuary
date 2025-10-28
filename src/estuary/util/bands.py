from enum import Enum

from estuary.util.constants import EIGHT_TO_4, FALSE_COLOR_4, FALSE_COLOR_8, RGB_4, RGB_8


class Bands(Enum):
    RGB = "rgb"
    FALSE_COLOR = "false_color"
    EIGHT = "8"
    FOUR = "4"

    def eight_band_idxes(self) -> tuple[int, ...]:
        if self == Bands.FALSE_COLOR:
            return FALSE_COLOR_8
        elif self == Bands.RGB:
            return RGB_8
        elif self == Bands.EIGHT:
            return tuple(range(8))
        elif self == Bands.FOUR:
            return (7, 5, 3, 1)
        else:
            raise RuntimeError(f"Unexpected band type {self}")

    def num_channels(self):
        if self in [Bands.FALSE_COLOR, Bands.RGB]:
            return 3
        elif self == Bands.EIGHT:
            return 8
        elif self == Bands.FOUR:
            return 4
        else:
            raise RuntimeError(f"Unexpected band type {self}")

    def band_order(self, inpt_channels: int) -> tuple[int, ...]:
        if self == Bands.FALSE_COLOR:
            if inpt_channels == 4:
                return FALSE_COLOR_4
            else:
                return FALSE_COLOR_8

        elif self == Bands.RGB:
            if inpt_channels == 4:
                return RGB_4
            else:
                return RGB_8

        elif self == Bands.FOUR:
            if inpt_channels == 4:
                return tuple(range(4))
            else:
                return EIGHT_TO_4

        elif self == Bands.EIGHT:
            if inpt_channels == 4:
                return EIGHT_TO_4
            else:
                return tuple(range(8))
        else:
            raise RuntimeError(f"Unexpected band type {self}")
