"""
todo:
cursor breaks if first lines are new lines

"""

xx = 1
if xx:
    from loguru import logger

import math
from copy import copy
from dataclasses import dataclass, field, asdict, fields, Field
from enum import StrEnum
from functools import lru_cache
from io import StringIO
from typing import Callable, Union, Any

import keyboard
import kivy.input
from kivy.app import App
from kivy.clock import mainthread, Clock
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout

Builder.load_string("""
<TextEdit>:
    canvas:
        Color:
            rgb : 0,0,0
        Line:
            rectangle : [0,0, self.width, self.height]
            
        Color:
            rgb: 1,0,0
        Rectangle:
            pos : self.cursor_x, self.cursor_y
            size : 2, self.cursor_height
        
""")

LB_NEWLINE: str = "\n"


class BBTagPRE(StrEnum):
    REF: str = "[ref={}]"
    BOLD: str = "[b]"
    ITALIC: str = "[i]"
    UNDERLINE: str = "[u]"
    STRIKETHROUGH: str = "[s]"
    FONT_NAME: str = "[font={}]"
    FONT_SIZE: str = "[size={}]"


class BBTagPOST(StrEnum):
    REF: str = "[/ref]"
    BOLD: str = "[/b]"
    ITALIC: str = "[/i]"
    UNDERLINE: str = "[/u]"
    STRIKETHROUGH: str = "[/s]"
    FONT_NAME: str = "[/font]"
    FONT_SIZE: str = "[/size]"


@lru_cache(64)
def get_extents(text: str, **kwargs) -> tuple[int, int]:
    return CoreLabel(**kwargs).get_extents(text)


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _get_string_unformatted(string: str) -> str:
    return string.encode("unicode_escape").decode("utf-8")


@dataclass
class CharSettings(object):
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    font_name: str = None
    font_size: int = None

    def copy(self) -> 'CharSettings':
        return copy(self)

    def get_active_tags(self) -> list[tuple[str, str], ...]:

        def check(dt_field: Field) -> bool:
            if dt_field.type is bool:
                return getattr(self, dt_field.name, False)

            elif dt_field.type is str:
                return isinstance(getattr(self, dt_field.name), str)

            elif dt_field.type is int:
                return isinstance(getattr(self, dt_field.name), int)

            logger.error(f"Property went unchecked. {dt_field.name, dt_field.type}")
            return False

        def adjust(property_name: str) -> tuple[str, str]:
            pre_tag: str = BBTagPRE[property_name]
            post_tag: str = BBTagPOST[property_name]

            if "{}" in pre_tag:
                pre_tag = pre_tag.format(getattr(self, property_name.lower()))

            if "{}" in post_tag:
                post_tag = post_tag.format(getattr(self, property_name.lower()))

            return pre_tag, post_tag

        return [adjust(unchecked_field.name.upper()) for unchecked_field in fields(self) if check(unchecked_field)]

    def get_wrapper(self) -> tuple[StringIO, StringIO]:
        pre_string: StringIO = StringIO()
        post_string: StringIO = StringIO()
        tag: tuple[str, str]
        for pre, post in self.get_active_tags():
            pre_string.write(pre)
            post_string.write(post)
        return pre_string, post_string

    def wrap(self, text: str) -> str:
        pre_string: StringIO
        post_string: StringIO
        pre_string, post_string = self.get_wrapper()
        return pre_string.getvalue() + text + post_string.getvalue()


@dataclass
class Character(object):
    text: str
    index: int
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    settings: CharSettings = field(default_factory=lambda: CharSettings())

    def __post_init__(self) -> None:
        pass

    def _refresh(self) -> None:
        logger.warning("Do not use this method.")
        self.width, self.height = get_extents(self.text, **asdict(self.settings))

    def set_position(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def set_size(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def get_bb_string(self) -> str:
        build_string: StringIO = StringIO()

        build_string.write(BBTagPRE.REF.format(self.index))

        pre_string: StringIO
        post_string: StringIO
        pre_string, post_string = self.settings.get_wrapper()
        build_string.write(pre_string.getvalue())
        build_string.write(self.text)
        build_string.write(post_string.getvalue())

        build_string.write(BBTagPOST.REF)

        return build_string.getvalue()

    def is_newline(self) -> bool:
        return self.text == LB_NEWLINE


@dataclass
class Cursor(object):
    x: int = 0
    y: int = 0

    def copy(self) -> 'Cursor':
        return Cursor(self.x, self.y)

    def __iter__(self):
        return iter((self.x, self.y))

    def set(self, cursor: 'Cursor') -> None:
        self.x = cursor.x
        self.y = cursor.y

    def __lt__(self, other: 'Cursor') -> bool:
        if self.y < other.y:
            return True
        return self.x < other.x if self.y == other.y else False

    def __le__(self, other: 'Cursor') -> bool:
        if self.y < other.y:
            return True

        return self.x <= other.x if self.y == other.y else False

    def __gt__(self, other: 'Cursor') -> bool:
        if self.y > other.y:
            return True
        return self.x > other.x if self.y == other.y else False

    def __ge__(self, other: 'Cursor') -> bool:
        if self.y > other.y:
            return True

        return self.x >= other.x if self.y == other.y else False


class _MarkupTextManager(object):
    def __init__(self, label: Label, refresh_callback: Callable):
        self._label: Label = label
        self._label.markup = True
        self._label.bind(refs=self._on_label_ref_updated)
        self._label.bind(texture=self._on_label_property_changed_wrapper)
        self._label.bind(texture_size=self._on_label_property_changed_wrapper)
        self._label.bind(size=self._on_label_property_changed_wrapper)
        self._refresh_callback: Callable = refresh_callback
        self._markup_text: str = ""
        self._characters: list[Character] = []
        self._lines: list[list[int]] = []
        self._cursor: Cursor = Cursor()
        self._global_settings: CharSettings = CharSettings()

        # multi select controls
        self._msc_initial_cursor: Cursor = Cursor()
        self._msc_end_cursor: Cursor = Cursor()
        self._msc_boxes: list[list[float, float, float, float]] = []

    def _on_label_property_changed_wrapper(self, *_args, **_kwargs) -> None:
        """
        Wrapper to simplify event binging
        :param _args: args
        :param _kwargs: kwargs
        :return: None
        """
        self._rebuild_lines()
        self._rebuild_selection_boxes()
        self.update()

    def get_unformatted_markup(self) -> str:
        return _get_string_unformatted(self._markup_text)

    def _get_text(self) -> str:
        return "".join(char.text for char in self._characters)

    def get_unformatted_text(self) -> str:
        return _get_string_unformatted(self._get_text())

    def _get_x(self, character: Character) -> float:
        return self._label.center_x - 0.5 * self._label.texture_size[0] + character.x

    def _get_y(self, character: Character) -> float:
        return self._label.center_y + 0.5 * self._label.texture_size[1] - character.y

    def _get_adjusted_y(self, character: Character) -> float:
        return self._get_y(character) - character.height

    def _rebuild_lines(self) -> None:
        """
        rebuild lines by y positions
        :return: None
        """
        self._lines.clear()

        current_line: list[int] = []

        # find all different y values to determine line count
        last_y: int = -1
        character: Character
        for character in self._characters:
            if character.y > last_y:
                current_line = []
                self._lines.append(current_line)
                last_y = character.y
            current_line.append(character.index)
        logger.debug(f"Lines ({len(self._lines)}): {self._lines}")

        if not self._is_cursor_valid():
            logger.debug("Cursor is invalid. Recalculating.")
            cursor: Cursor = self._get_next_valid_cursor()
            logger.debug(f"{self._cursor} -> {cursor}")
            self._set_cursor(cursor)

    def _get_next_valid_cursor(self) -> Cursor:
        """
        get next valid cursor
        :return: Cursor object
        """
        cursor: Cursor = self._cursor.copy()

        # if row is invalid get next closest row
        if self._cursor.y >= len(self._lines):
            cursor.y = len(self._lines) - 1
            cursor.x = len(self._lines[cursor.y])

        # if column is invalid get next closest
        elif self._cursor.x > len(self._lines[cursor.y]):
            if cursor.y + 1 < len(self._lines):
                cursor.y += 1
                cursor.x = 0

            else:
                cursor.x = len(self._lines[cursor.y]) - 1

        if not self._is_cursor_valid(cursor):
            max_x: list[int] = [len(line) for line in self._lines]
            logger.error(f"Generated cursor is not valid: {cursor} | [max_x={max_x}], max_y={len(self._lines)}]")

        return cursor

    def _is_cursor_valid(self, cursor: Cursor = None) -> bool:
        """
        check if cursor is valid
        :return: None
        """
        if cursor is None:
            cursor = self._cursor

        if cursor.y == 0 and cursor.x == 0:
            return True

        if cursor.y >= len(self._lines) or cursor.y < 0:
            return False

        line: list[int] = self._lines[cursor.y]
        return len(line) >= cursor.x >= 0

    def _get_line_of_index(self, index: int) -> list[int]:
        return next((line for line in self._lines if index in line), [])

    def _get_index_by_cursor(self) -> int:
        """
        return index cursor is pointing at
        :return: index
        """
        if self._cursor.y >= len(self._lines):
            logger.warning("Cannot get index: Row out of range.")
            return -1

        line: list[int] = self._lines[0]

        x: int = self._cursor.x
        if x >= len(line):
            logger.warning("Cannot get index: Column out of range.")
            return -1

        return line[x]

    def _on_label_ref_updated(self, _widget: Label, refs: dict[str, list[tuple[int, int, int, float], ...]]) -> None:
        """
        Called if the label updates its ref data
        :param _widget: label
        :param refs: ref dictionary
        :return: None
        """
        logger.debug("")
        logger.debug(f"Text: {self.get_unformatted_text()}")
        logger.debug(f"Markup: {self.get_unformatted_markup()}")
        logger.debug(f"Updating refs: {refs}")
        # update character positions and size
        index_list: list[int] = [character.index for character in self._characters]
        str_index: str
        ref_data: list[tuple[int, int, int, float], ...]

        for str_index, ref_data in refs.items():
            index: int = int(str_index)
            data: tuple[int, int, int, float] = ref_data[0]
            x: int = data[0]
            y: int = data[1]
            width: int = data[2] - x
            height: float = data[3] - y
            int_height: int = int(height)
            if height > int_height:
                logger.warning(f"{str_index}: Height is a float {height}")
            character: Character = self._characters[index]
            character.set_position(x, y)
            character.set_size(width, int_height)
            index_list.remove(index)

        # adjust characters with zero width and missing ones
        missing_index: int
        is_previous: bool = True
        for missing_index in index_list:
            pivot_index: int = missing_index

            if pivot_index == 0:
                is_previous = False
                while pivot_index in index_list:
                    pivot_index += 1
                    if pivot_index > len(self._characters):
                        return logger.error("Text must not contain only missing characters.")
            else:
                pivot_index -= 1

            missing_character: Character = self._characters[missing_index]
            pivot_character: Character = self._characters[pivot_index]

            logger.debug(f"Adjusting {missing_character} with {pivot_character}.")

            if is_previous:
                missing_character.x = pivot_character.x + pivot_character.width
            else:
                missing_character.x = pivot_character.x - pivot_character.width

            missing_character.y = pivot_character.y
            missing_character.height = pivot_character.height
            missing_character.width = pivot_character.width

        self._rebuild_lines()
        self.update()

    def _format(self, text: str, character_settings: list[Character] = None) -> None:
        """
        Format and process text.
        :param text: text to format
        :param character_settings: list of character settings to use. Require to be same size as text.
        :return: None
        """
        self._characters.clear()
        # fill refs for each character
        char: str
        index: int
        build_string: StringIO = StringIO()
        for index, char in enumerate(text):
            character: Character = Character(
                char,
                index,
                settings=self._global_settings.copy() if character_settings is None else character_settings[index]
            )
            build_string.write(character.get_bb_string())
            self._characters.append(character)
        self._markup_text = build_string.getvalue()
        logger.debug(f"MarkupText: {self.get_unformatted_markup()}")
        self._label.text = self._markup_text

    def update(self) -> None:
        """
        call refresh hook
        :return: None
        """
        self._refresh_callback(
            self.get_cursor_pos(),
            self.get_cursor_size(),
            self._msc_boxes
        )

    def update_deferred(self, timeout: Union[int, float] = 0) -> None:
        """
        call refresh hook after given time
        :return: None
        """
        Clock.schedule_once(lambda _dt: self.update(), timeout)

    def set_text(self, text: str) -> None:
        """
        Set new text. Forces complete recalculation.
        :param text: text to set
        :return: None
        """
        self._format(text)

    def get_cursor_size(self) -> tuple[float, float]:
        """
        get cursor size by current row and column
        :return: width, height
        """
        character: Character
        current_line: list[int] = self._lines[self._cursor.y]

        if self._cursor.x >= len(current_line):
            character = self._characters[current_line[-1]]
        elif self._cursor.x > 0:
            character = self._characters[current_line[self._cursor.x - 1]]
        else:
            character = self._characters[current_line[0]]

        return character.width, character.height

    def get_cursor_pos(self) -> tuple[float, float]:
        """
        get cursor position by the current row and column
        :return: x,y position
        """
        if self._cursor.y >= len(self._lines):
            logger.warning(f"Cursor y position out of bounds: {self._cursor.y}")
            return 0, 0

        current_line: list[int] = self._lines[self._cursor.y]

        if not current_line:
            return self._cursor.x, self._cursor.y

        # find max character height in line to adjust y
        max_height = max(self._characters[index].height for index in current_line)
        pivot_character: Character = self._characters[
            current_line[min(max(self._cursor.x - 1, 0), len(current_line) - 1)]
        ]
        character: Character = self._characters[current_line[min(self._cursor.x, len(current_line) - 1)]]
        adjusted: float = 0

        if max_height > pivot_character.height:
            adjusted = (max_height - pivot_character.height) / 2.0 + 1

        x: float = self._get_x(character) + (character.width if self._cursor.x >= len(current_line) else 0)
        y: float = self._get_adjusted_y(pivot_character) - adjusted

        return x, y

    def set_cursor_by_touch(self, touch: kivy.input.MotionEvent) -> None:
        """
        set cursor by touch position
        :param touch: touch
        :return: None
        """
        self._set_cursor(self._get_closest_cursor_to_pos(touch.x, touch.y))
        self.update()

    def _set_cursor(self, cursor: Cursor) -> None:
        """
        set current cursor by given cursor
        :param cursor: new cursor
        :return: None
        """
        logger.debug(f"Setting cursor {cursor}")
        self._cursor.set(cursor)

        if not self._is_cursor_valid() and (self._cursor.x != 0 or self._cursor.y != 0):
            logger.warning("Given cursor is invalid.")
            self._set_cursor(self._get_next_valid_cursor())

    def _set_cursor_pos(self, x: int, y: int) -> None:
        """
        set cursor row and column by index
        :param x: column
        :param y: row
        :return: None
        """
        self._cursor.x, self._cursor.y = x, y

    def _get_cursor_by_index(self, index: int) -> Cursor:
        """
        find row , column by given index
        :param index: int index
        :return: tuple of row column
        """
        line: list[int]
        for line_index, line in enumerate(self._lines):
            if index in line:
                return Cursor(line.index(index), line_index)
        return Cursor(-1, -1)

    def _get_closest_cursor_to_pos(self, x: float, y: float) -> Cursor:
        """
        get the closest index to x, y position
        :param x: x pos
        :param y: y pos
        :return: row, col
        """
        if len(self._characters) == 0:
            return Cursor(-1, -1)

        x, y = self._label.to_local(x, y)
        closest_index: int = 0
        min_distance: float = math.inf
        target_point: tuple[float, float] = (x, y)
        character: Character
        for character in self._characters:
            other_point: tuple[float, float] = (self._get_x(character), self._get_y(character) - character.height / 2.0)
            ecd: float = euclidean_distance(target_point, other_point)
            if ecd < min_distance:
                min_distance = ecd
                closest_index = character.index

        line: list[int] = self._get_line_of_index(closest_index)
        line_index: int = self._lines.index(line)
        character_line_index: int = line.index(closest_index)
        if closest_index == line[-1]:
            closest_character: Character = self._characters[closest_index]
            if min_distance > closest_character.height / 2.0:
                character_line_index += 1
        return Cursor(character_line_index, line_index)

    def move_cursor_right(self) -> None:
        """
        move cursor right
        :return: None
        """
        current_line: list[int] = self._lines[self._cursor.y]

        if self._cursor.x + 1 > len(current_line):
            if self._cursor.y + 1 < len(self._lines):
                self._cursor.x = 0
                self._cursor.y += 1
        else:
            self._cursor.x += 1

        self.update()

    def move_cursor_left(self) -> None:
        """
        move cursor left
        :return: None
        """
        if self._cursor.x < 1:
            if self._cursor.y >= 1:
                self._cursor.y -= 1
                self._cursor.x = len(self._lines[self._cursor.y])
        else:
            self._cursor.x -= 1

        self.update()

    def move_cursor_up(self) -> None:
        """
        move cursor up
        :return: None
        """
        self._cursor.y = max(self._cursor.y - 1, 0)
        current_line: list[int] = self._lines[self._cursor.y]
        self._cursor.x = min(self._cursor.x, len(current_line))

        self.update()

    def move_cursor_down(self) -> None:
        """
        move cursor down
        :return: None
        """
        self._cursor.y = min(self._cursor.y + 1, len(self._lines) - 1)
        current_line: list[int] = self._lines[self._cursor.y]
        self._cursor.x = min(self._cursor.x, len(current_line))

        self.update()

    def start_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        """
        start selection by drag
        :param touch: touch event
        :return: None
        """
        self._msc_initial_cursor = self._get_closest_cursor_to_pos(touch.x, touch.y)
        self._msc_end_cursor = self._msc_initial_cursor

    def update_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        """
        update multi select drag
        :param touch: touch event
        :return: None
        """
        self._msc_end_cursor = self._get_closest_cursor_to_pos(touch.x, touch.y)
        self._rebuild_selection_boxes()
        self._set_cursor(self._msc_end_cursor)

    def stop_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        """
        stop multi select drag
        :param touch: touch event
        :return: None
        """
        if self._is_multiselect():
            self._rebuild_selection_boxes()
            self._set_cursor(self._msc_end_cursor)
        else:
            self._msc_boxes.clear()
            self.set_cursor_by_touch(touch)

    def _cursor_to_index(self, cursor: Cursor) -> int:
        """
        get index at cursor position
        :param cursor: cursor object
        :return: index, returns -1 if cursor column (x) is out of bounds
        """
        line: list[int] = self._lines[cursor.y]
        return line[cursor.x] if cursor.x < len(line) else -1

    def _is_multiselect(self) -> bool:
        return self._msc_initial_cursor != self._msc_end_cursor

    def _get_selected_indices(self) -> list[int]:
        """
        get selected indices
        :return: list containing all selected indices
        """
        start_cursor: Cursor = min(self._msc_initial_cursor, self._msc_end_cursor)
        end_cursor: Cursor = max(self._msc_initial_cursor, self._msc_end_cursor)

        # fill indices
        start_index: int = self._cursor_to_index(start_cursor)
        end_index: int = self._cursor_to_index(end_cursor)

        # get e index
        e_start_index: int = start_index if start_index >= 0 else self._lines[start_cursor.y][-1]
        e_end_index: int = end_index if end_index >= 0 else self._lines[end_cursor.y][-1]

        return list(
            range(e_start_index + (1 if start_index < 0 else 0), e_end_index + (1 if end_index < 0 else 0))
        )

    def _get_selected_characters(self) -> list[Character]:
        """
        return a list of all selected characters
        :return: list of characters
        """
        return [self._characters[index] for index in self._get_selected_indices()]

    def _set_char_setting(self, name: str, value: Any) -> None:
        """
        set char setting
        :param name: name of the setting
        :param value: value
        :return: None
        """
        for character in self._get_selected_characters():
            setattr(character.settings, name, value)

        self._rebuild_from_characters()

    def set_bold(self, value: bool) -> None:
        """
        set bold value
        :param value: new bold state
        :return: None
        """
        self._set_char_setting("bold", value)

    def set_italic(self, value: bool) -> None:
        """
        set italic value
        :param value: new italic state
        :return: None
        """
        self._set_char_setting("italic", value)

    def set_underline(self, value: bool) -> None:
        """
        set underline value
        :param value: new underline state
        :return: None
        """
        self._set_char_setting("underline", value)

    def set_strikethrough(self, value: bool) -> None:
        """
        set strikethrough value
        :param value: new strikethrough state
        :return: None
        """
        self._set_char_setting("strikethrough", value)

    def set_font_name(self, value: str) -> None:
        """
        set font name value
        :param value: new font name state
        :return: None
        """
        self._set_char_setting("font_name", value)

    def set_font_size(self, value: int) -> None:
        """
        set font size value
        :param value: new font size state
        :return: None
        """
        self._set_char_setting("font_size", value)

    def _rebuild_selection_boxes(self) -> None:
        """
        rebuild selection boxes.
        Boxes are per-line rectangles in x,y,w,h format
        :return: None
        """
        self._msc_boxes.clear()

        if not self._is_multiselect():
            return self.update()

        span_indices: list[int] = self._get_selected_indices()

        # split indices into lines
        lines: list[list[int]] = [[] for _ in self._lines]
        for index in span_indices:
            for line_index, line in enumerate(self._lines):
                if index in line:
                    lines[line_index].append(index)
                    break

        # find max height of each line
        line_heights: list[int] = []

        for line in lines:
            max_height: int = 0
            for char_index in line:
                character: Character = self._characters[char_index]
                max_height = max(character.height, max_height)

            line_heights.append(max_height)

        # calculate boxes
        for line_index, line in enumerate(lines):
            if not line:
                continue

            rectangle: list[float, float, float, float] = [0.0, 0.0, 0.0, 0.0]
            first_char: Character = self._characters[line[0]]
            last_char: Character = self._characters[line[-1]]
            rectangle[0] = self._get_x(first_char)
            rectangle[1] = self._get_y(last_char) - line_heights[line_index]
            rectangle[2] = self._get_x(last_char) - rectangle[0] + last_char.width
            rectangle[3] = line_heights[line_index]

            self._msc_boxes.append(rectangle)

        self.update()

    def _rebuild_from_characters(self) -> None:
        """
        rebuild from characters
        :return: None
        """
        build_string: StringIO = StringIO()
        index: int
        character: Character
        for index, character in enumerate(self._characters):
            character.index = index
            build_string.write(character.get_bb_string())
        self._markup_text = build_string.getvalue()
        logger.debug(f"MarkupText: {self.get_unformatted_markup()}")
        self._label.text = self._markup_text

    def _reset_selection(self) -> None:
        """
        reset selection parameters
        :return: None
        """
        self._msc_end_cursor = self._msc_initial_cursor
        self._rebuild_selection_boxes()

    def delete_selected(self) -> None:
        """
        delete selected text
        :return: None
        """
        selected_indices: list[int] = self._get_selected_indices()
        for character in self._characters.copy():
            if character.index in selected_indices:
                self._characters.remove(character)
        self._rebuild_from_characters()
        self._rebuild_lines()
        self._reset_selection()

    def _multi_delete(self) -> None:
        """
        multi delete
        :return: None
        """
        self.delete_selected()
        self._set_cursor(min(self._msc_initial_cursor, self._msc_end_cursor))

    def delete(self, left: bool = True) -> None:
        """
        delete char at current position
        :param left: delete on left or right side
        :return: None
        """
        if self._is_multiselect():
            self._multi_delete()
        else:
            delete_index: int = self._cursor_to_index(self._cursor)

            if delete_index == -1:
                delete_index = self._cursor_to_index(Cursor(self._cursor.x - 1, self._cursor.y))

            elif left:
                delete_index -= 1

            delete_index = max(delete_index, 0)

            del self._characters[delete_index]

            self._rebuild_from_characters()
            self._rebuild_lines()

            self.move_cursor_left()

        self.update_deferred()

    def insert(self, text: str, settings: CharSettings = None) -> None:
        """
        insert text at cursor position
        :param text: text to insert
        :param settings: text settings
        :return: None
        """
        # store cursor
        start_cursor: Cursor = self._cursor.copy()
        # get index cursor is currently at
        start_index: int = self._cursor_to_index(start_cursor)
        # controls if text should be inserted at index or after
        # if 'insert_at' is True new characters are inserted before the current element at that index
        # [a, b, c] -> insert at 1 -> [a, x, b, c]
        # if 'insert_at' is False new characters are inserted after the current element at that index
        # [a, b, c] -> insert at 1 -> [a, b, x, c]
        insert_at: bool = True

        # if cursor is out of bounds get index of next smaller cursor
        if start_index == -1:
            start_index = self._cursor_to_index(Cursor(start_cursor.x - 1, start_cursor.y))
            insert_at = False

        # remove selected chars if multi selected
        is_multiselect: bool = self._is_multiselect()

        if is_multiselect:
            start_cursor = min(self._msc_initial_cursor, self._msc_end_cursor)
            start_index = max(self._cursor_to_index(start_cursor), 0)
            self.delete_selected()
            self._set_cursor(start_cursor)

        char: str
        character: Character
        for index, char in enumerate(text):
            insert_index: int = index + start_index + (0 if insert_at else 1)
            character = Character(char, insert_index, settings=settings or self._global_settings.copy())
            self._characters.insert(insert_index, character)

        self._rebuild_from_characters()
        self._rebuild_lines()

        for _ in text:
            self.move_cursor_right()

        self.update_deferred()


class TextEdit(RelativeLayout):
    cursor_x = NumericProperty(0)
    cursor_y = NumericProperty(0)
    cursor_height = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        lbl = Label(text="Hello World\n12345678", color=(0, 0, 0))
        lbl.text_size = (51, None)
        self._gfx_selection = []
        self.add_widget(lbl)
        self._manager: _MarkupTextManager = _MarkupTextManager(lbl, self._cb)
        self._manager.set_text(lbl.text)

        keyboard.add_hotkey("left arrow", lambda *args: self._manager.move_cursor_left())
        keyboard.add_hotkey("right arrow", lambda *args: self._manager.move_cursor_right())
        keyboard.add_hotkey("up arrow", lambda *args: self._manager.move_cursor_up())
        keyboard.add_hotkey("down arrow", lambda *args: self._manager.move_cursor_down())

        def insert_text(kbe: keyboard.KeyboardEvent):
            if kbe.name in keyboard.all_modifiers or kbe.name in ["nach-oben", "nach-unten", "nach-rechts",
                                                                  "nach-links"]:
                return

            s = kbe.name

            if kbe.name == "enter":
                s = "\n"

            if kbe.name == "space":
                s = " "

            if kbe.name == "backspace":
                self._manager.delete()
                return

            self._manager.insert(s)

        keyboard.on_release(insert_text)

    @mainthread
    def _cb(self, pos, size, boxes):
        self.cursor_x = pos[0]
        self.cursor_y = pos[1]
        self.cursor_height = size[1]

        for instruction in self._gfx_selection:
            self.canvas.remove(instruction)
        self._gfx_selection.clear()

        for box in boxes:
            with self.canvas:
                c = Color(1, 1, 0, 0.5)
                r = Rectangle(pos=(box[0], box[1]), size=(box[2], box[3]))
                self._gfx_selection.append(c)
                self._gfx_selection.append(r)

    def on_touch_down(self, touch):
        touch.push()
        touch.apply_transform_2d(self.to_local)
        self._manager.start_select_by_drag(touch)
        touch.pop()

    def on_touch_move(self, touch):
        touch.push()
        touch.apply_transform_2d(self.to_local)
        self._manager.update_select_by_drag(touch)
        touch.pop()

    def on_touch_up(self, touch: kivy.input.MotionEvent):
        touch.push()
        touch.apply_transform_2d(self.to_local)
        self._manager.stop_select_by_drag(touch)
        touch.pop()


if __name__ == "__main__":
    class Root(FloatLayout):
        def on_kv_post(self, base_widget):
            self.add_widget(TextEdit(size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.5}))


    class TextEditApp(App):
        def build(self):
            Window.clearcolor = (1, 1, 1, 1)
            return Root()


    TextEditApp().run()
