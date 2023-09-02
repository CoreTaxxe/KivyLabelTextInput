#
import math
from copy import copy
from typing import Callable, Union

if 1 == 1:
    from loguru import logger
#
from dataclasses import dataclass, field, asdict, fields, Field
from enum import StrEnum
from functools import lru_cache
import keyboard
from io import StringIO

from kivy.lang import Builder
import kivy.input
from kivy.app import App
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.clock import mainthread, Clock
from kivy.properties import NumericProperty
from kivy.graphics import Rectangle, Color

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


class _MarkupTextManager(object):
    def __init__(self, label: Label, refresh_callback: Callable):
        self._label: Label = label
        self._label.markup = True
        self._label.bind(refs=self._on_label_ref_updated)
        self._label.bind(texture=self._on_label_property_changed_wrapper)
        self._label.bind(size=self._on_label_property_changed_wrapper)
        self._refresh_callback: Callable = refresh_callback
        self._text: str = ""
        self._markup_text: str = ""
        self._characters: list[Character] = []
        self._lines: list[list[int]] = []
        self._cursor: list[int, int] = [0, 0]
        self._global_settings: CharSettings = CharSettings()

        # multi select controls
        self._msc_initial_index: int = 0
        self._msc_end_index: int = 0
        self._msc_boxes: list[list[float, float, float, float]] = []
        # line break control
        self._lbc_last_indices: list[int] = []

    def _on_label_property_changed_wrapper(self, *_args, **_kwargs) -> None:
        """
        Wrapper to simplify event binging
        :param _args: args
        :param _kwargs: kwargs
        :return: None
        """
        self.update()

    def get_unformatted_markup(self) -> str:
        return _get_string_unformatted(self._markup_text)

    def get_unformatted_text(self) -> str:
        return _get_string_unformatted(self._text)

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
        self._lbc_last_indices.clear()

        current_line: list[int] = []

        # find all different y values to determine line count
        last_y: int = -1
        character: Character
        last_character: Union[Character, None] = None
        new_line_index: int = -1
        for character in self._characters:
            if character.y > last_y:
                current_line = []
                self._lines.append(current_line)
                last_y = character.y
                if last_character is not None and last_character.text != LB_NEWLINE:
                    self._lbc_last_indices.append(new_line_index)
                    new_line_index -= 1
            current_line.append(character.index)
            last_character = character
        logger.debug(f"Lines ({len(self._lines)}): {self._lines}")

    def _get_line_of_index(self, index: int) -> list[int]:
        return next((line for line in self._lines if index in line), [])

    def _get_index_by_cursor(self) -> int:
        """
        return index cursor is pointing at
        :return: index
        """
        if self._cursor[1] >= len(self._lines):
            logger.warning("Cannot get index: Row out of range.")
            return -1

        line: list[int] = self._lines[0]

        x: int = self._cursor[0]
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

    def _format(self) -> None:
        """
        Format and process text
        :return: None
        """
        self._characters.clear()
        # fill refs for each character
        char: str
        index: int
        build_string: StringIO = StringIO()
        for index, char in enumerate(self._text):
            character: Character = Character(char, index, settings=self._global_settings.copy())
            build_string.write(character.get_bb_string())
            self._characters.append(character)
        self._markup_text = build_string.getvalue()
        logger.debug(f"MarkupText: {self._markup_text}")
        self._label.text = self._markup_text

    def update(self) -> None:
        self._refresh_callback(
            self.get_cursor_pos(),
            self.get_cursor_size(),
            self._msc_boxes
        )

    def set_text(self, text: str) -> None:
        """
        Set new text. Forces complete recalculation.
        :param text: text to set
        :return: None
        """
        self._text = text
        self._format()

    def get_cursor_size(self) -> tuple[float, float]:
        """
        get cursor size by current row and column
        :return: width, height
        """
        character: Character
        current_line: list[int] = self._lines[self._cursor[1]]

        if self._cursor[0] >= len(current_line):
            character = self._characters[current_line[-1]]
        elif self._cursor[0] > 0:
            character = self._characters[current_line[self._cursor[0] - 1]]
        else:
            character = self._characters[current_line[0]]

        return character.width, character.height

    def get_cursor_pos(self) -> tuple[float, float]:
        """
        get cursor position by the current row and column
        :return: x,y position
        """
        if self._cursor[1] >= len(self._lines):
            logger.warning(f"Cursor y position out of bounds: {self._cursor[1]}")
            return 0, 0

        character: Character
        current_line: list[int] = self._lines[self._cursor[1]]

        # if cursor x is bigger than the size of the line get the most right element pos
        if self._cursor[0] >= len(current_line):
            character = self._characters[current_line[-1]]
            return self._get_x(character) + character.width, self._get_adjusted_y(character)
        else:
            character = self._characters[current_line[self._cursor[0]]]
            return self._get_x(character), self._get_adjusted_y(character)

    def set_cursor_by_touch(self, touch: kivy.input.MotionEvent) -> None:
        """
        set cursor by touch position
        :param touch: touch
        :return: None
        """
        self._set_cursor_by_index(self._get_closest_index_to_pos(touch.x, touch.y))
        self.update()

    def _set_cursor_by_index(self, index: int) -> None:
        """
        set cursor row and column by index
        :param index: int index
        :return: None
        """
        self._cursor[0], self._cursor[1] = self._get_cursor_by_index(index)

    def _get_cursor_by_index(self, index: int) -> tuple[int, int]:
        """
        find row , column by given index
        :param index: int index
        :return: tuple of row column
        """

        # index is new line
        if index < 0:
            line_index: int = (index + 1) * -1
            return len(self._lines[line_index]), line_index

        line: list[int]
        for line_index, line in enumerate(self._lines):
            if index in line:
                return line.index(index), line_index
        return 0, 0

    def _get_closest_index_to_pos(self, x: float, y: float) -> int:
        """
        get the closest index to x, y position
        :param x: x pos
        :param y: y pos
        :return: index, returns -1 if no index was found
        """
        if len(self._characters) == 0:
            return -1

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
        if closest_index == line[-1]:
            closest_character: Character = self._characters[closest_index]
            if min_distance > closest_character.height / 2.0:
                closest_index = (self._lines.index(line) + 1) * -1
        return closest_index

    def move_cursor_right(self) -> None:
        """
        move cursor right
        :return: None
        """
        current_line: list[int] = self._lines[self._cursor[1]]

        if self._cursor[0] + 1 > len(current_line):
            if self._cursor[1] + 1 < len(self._lines):
                self._cursor[0] = 0
                self._cursor[1] += 1
        else:
            self._cursor[0] += 1

        self.update()

    def move_cursor_left(self) -> None:
        """
        move cursor left
        :return: None
        """
        if self._cursor[0] < 1:
            if self._cursor[1] >= 1:
                self._cursor[1] -= 1
                self._cursor[0] = len(self._lines[self._cursor[1]])
        else:
            self._cursor[0] -= 1

        self.update()

    def move_cursor_up(self) -> None:
        """
        move cursor up
        :return: None
        """
        self._cursor[1] = max(self._cursor[1] - 1, 0)
        current_line: list[int] = self._lines[self._cursor[1]]
        self._cursor[0] = min(self._cursor[0], len(current_line))

        self.update()

    def move_cursor_down(self) -> None:
        """
        move cursor down
        :return: None
        """
        self._cursor[1] = min(self._cursor[1] + 1, len(self._lines) - 1)
        current_line: list[int] = self._lines[self._cursor[1]]
        self._cursor[0] = min(self._cursor[0], len(current_line))

        self.update()

    def start_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        """
        start selection by drag
        :param touch: touch event
        :return: None
        """
        self._msc_initial_index = self._get_closest_index_to_pos(touch.x, touch.y)
        self._msc_end_index = self._msc_initial_index

    def update_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        """
        update multi select drag
        :param touch: touch event
        :return: None
        """
        self._msc_end_index = self._get_closest_index_to_pos(touch.x, touch.y)
        self._rebuild_selection_boxes()

    def stop_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        if self._msc_initial_index == self._msc_end_index:
            self._msc_boxes.clear()
            self.set_cursor_by_touch(touch)
        else:
            self._rebuild_selection_boxes()

    def _rebuild_selection_boxes(self) -> None:
        """
        rebuild selection boxes.
        Boxes are per-line rectangles in x,y,w,h format
        :return: None
        """
        self._msc_boxes.clear()

        self._set_cursor_by_index(self._msc_end_index)

        if self._msc_initial_index == self._msc_end_index:
            return self.update()

        start_index: int = self._msc_initial_index
        end_index: int = self._msc_end_index

        e_end_index: int = min(start_index, end_index)

        if start_index < 0:
            start_index = self._lines[(start_index + 1) * -1][-1]

        if end_index < 0:
            end_index = self._lines[(end_index + 1) * -1][-1]

        start_index, end_index = sorted([start_index, end_index])

        # fill indices
        indices: list[int] = list(range(start_index, end_index + 1))

        print()
        print("auto NL: ", self._lbc_last_indices)
        print(indices)
        print([self._characters[c].text for c in indices])

        # split indices into lines
        lines: list[list[int]] = [[] for _ in self._lines]
        for index in indices:
            for line_index, line in enumerate(self._lines):
                if index in line:
                    lines[line_index].append(index)
                    break

        # find indices for row spanning e indices
        e_indices: set[int] = set()
        prev_line: Union[list[int], None] = None
        for line_index, line in enumerate(lines):
            if not line:
                continue

            if prev_line is not None:
                e_indices.add(line_index * -1)

            elif e_end_index < 0:
                e_indices.add((line_index + 1) * -1)

            prev_line = line

        # convert to list and sort
        e_indices: list[int] = sorted(list(e_indices))
        print("EE", e_indices)

        # find max height of each line
        line_height_map: list[int] = []

        for line in lines:
            max_height: int = 0
            for char_index in line:
                character: Character = self._characters[char_index]
                max_height = max(character.height, max_height)

            line_height_map.append(max_height)

        for line_index, line in enumerate(lines):
            if not line:
                continue

            rectangle: list[float, float, float, float] = [0.0, 0.0, 0.0, 0.0]
            first_char: Character = self._characters[line[0]]
            last_char: Character = self._characters[line[-1]]
            rectangle[0] = self._get_x(first_char)
            rectangle[1] = self._get_y(last_char) - line_height_map[line_index]
            rectangle[2] = self._get_x(last_char) - rectangle[0]
            rectangle[3] = line_height_map[line_index]

            print(line, self._lines[line_index])

            e_index: int = (line_index + 1) * -1

            print("EI, ", e_index)

            if e_index in e_indices:
                print("Extend ", e_indices)
                rectangle[2] += last_char.width

            self._msc_boxes.append(rectangle)

        self.update()


class TextEdit(RelativeLayout):
    cursor_x = NumericProperty(0)
    cursor_y = NumericProperty(0)
    cursor_height = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        lbl = Label(text="Hello World\n12345678", color=(0, 0, 0))
        lbl.text_size = (50, None)
        self._gfx_selection = []
        self.add_widget(lbl)
        self._manager: _MarkupTextManager = _MarkupTextManager(lbl, self._cb)
        self._manager.set_text(lbl.text)

        keyboard.add_hotkey("left arrow", lambda *args: self._manager.move_cursor_left())
        keyboard.add_hotkey("right arrow", lambda *args: self._manager.move_cursor_right())
        keyboard.add_hotkey("up arrow", lambda *args: self._manager.move_cursor_up())
        keyboard.add_hotkey("down arrow", lambda *args: self._manager.move_cursor_down())

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
