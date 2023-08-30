# todo double click to select word

import math
from io import StringIO
from typing import Callable, Union, Any

import keyboard
import kivy.input
from kivy.app import App
from kivy.clock import mainthread, Clock
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from loguru import logger

Builder.load_string("""
<TextEdit>:

    canvas:
        Color:
            rgb: 1,0,0
        Rectangle:
            pos : self.cursor_x, self.cursor_y
            size : 2, self.cursor_height


    canvas.after:
        Color:
            rgb : 0,0,0

        Line:
            rectangle: [self.x, self.y, self.width, self.height]
""")

PRE_REF: str = "[ref={}]"
POST_REF: str = "[/ref]"

PRE_BOLD: str = "[b]"
POST_BOLD: str = "[/b]"

PRE_ITALIC: str = "[i]"
POST_ITALIC: str = "[/i]"

PRE_UNDERLINE: str = "[u]"
POST_UNDERLINE: str = "[/u]"

PRE_STRIKETHROUGH: str = "[s]"
POST_STRIKETHROUGH: str = "[/s]"

PRE_FONT: str = "[font={}]"
POST_FONT: str = "[/font]"

PRE_FONT_SIZE: str = "[size={}]"
POST_FONT_SIZE: str = "[/size]"


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _get_string_unformatted(string: str) -> str:
    return string.encode("unicode_escape").decode("utf-8")


class _MarkupTextManager(object):
    DEFAULT_CHAR_DATA: dict[str, Union[bool, None, str, int, float]] = {
        "bold": False,
        "italic": False,
        "underline": False,
        "strikethrough": False,
        "font": None,
        "font_size": -1,
    }

    def __init__(self, label: Label, refresh_callback: Callable):
        self._refresh_callback: Callable = refresh_callback
        self._label: Label = label
        self._label.markup = True
        self._label.bind(refs=self._on_label_refs_updated)
        self._label.bind(texture=lambda *args: self.update())
        self._label.bind(texture_size=lambda *args: self.update())
        self._label.bind(size=lambda *args: self.update())
        self._label.bind(pos=lambda *args: self.update())
        self._label.bind(text_size=lambda *args: self.update())
        self._label.bind(halign=lambda *args: self.update())
        self._label.bind(valign=lambda *args: self.update())

        self._text: str = ""
        self._markup_text: str = ""
        self._lines: list[list] = [[]]
        self._selection_boxes: list[list[float, float, float, float]] = []
        self._selected_indices: list[int] = []
        self._link_indices: set[int] = set()
        self._initial_selected_index: int = 0
        self._char_map: dict[int, tuple[int, str]] = {}
        self._char_data_map: dict[int, dict[str, Union[bool, None, str, int, float]]] = {}
        self._new_char_data_map: dict[int, dict[str, Union[bool, None, str, int, float]]] = {}
        self._index_map: dict = {}
        self._current_index: int = 0

        self._format()

    def _get_char_at(self, index: int) -> str:
        if not isinstance(index, int):
            raise TypeError(f"Expected type {int} got {type(index)}")
        return self._char_map.get(index, (0, ""))[1]  # noqa

    def _get_line_at(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError(f"Expected type {int} got {type(index)}")
        return self._char_map.get(index, (len(self._lines) - 1, ""))[0]

    def _get_x(self, index: int) -> float:
        if not isinstance(index, int):
            raise TypeError(f"Expected type {int} got {type(index)}")
        return self._label.center_x - 0.5 * self._label.texture_size[0] + self._index_map.get(index, (0, 0, 0, 0))[0]

    def _get_y(self, index: int) -> float:
        if not isinstance(index, int):
            raise TypeError(f"Expected type {int} got {type(index)}")
        return self._label.center_y + self._label.texture_size[1] * 0.5 - self._index_map.get(index, (0, 0, 0, 0))[1]

    def _get_w(self, index: int) -> float:
        if not isinstance(index, int):
            raise TypeError(f"Expected type {int} got {type(index)}")
        _data: list[float, float, float, float] = self._index_map.get(index, (0, 0, 0, 0))
        return _data[2] - _data[0]

    def _get_h(self, index: int) -> float:
        if not isinstance(index, int):
            raise TypeError(f"Expected type {int} got {type(index)}")
        _data: list[float, float, float, float] = self._index_map.get(index, (0, 0, 0, 0))
        return _data[3] - _data[1]

    def _on_label_refs_updated(self, _widget: Label, values: dict[str, list[list[int]]]) -> None:
        self._index_map = {int(index): value[0] for index, value in values.items()}

        if not self._index_map:
            return self.update()

        # fix missing indices by assuming new lines
        _sorted_indices: list[int] = sorted(self._index_map.keys())

        for potentially_missing_index in range(len(self._text) + 1):
            if potentially_missing_index in _sorted_indices:
                continue

            # if index is missing, take previous index and extend it
            # cover potential case that first character is missing
            data: list[float, float, float, float]

            if potentially_missing_index == 0:
                data = self._index_map[_sorted_indices[0]]
                self._index_map[potentially_missing_index] = [data[0] + (data[2] - data[0]), data[1] - data[3], data[2],
                                                              data[3]]
            else:
                data = self._index_map[_sorted_indices[potentially_missing_index - 1]]
                self._index_map[potentially_missing_index] = [data[0] + (data[2] - data[0]), data[1], data[2], data[3]]

            _sorted_indices.insert(potentially_missing_index, potentially_missing_index)

        self._adjust_lines_by_autowrap()

        self.update()

    def _adjust_lines_by_autowrap(self) -> None:
        logger.error("Line autowrap not implemented")

        for line in self._lines:
            pass

    def _set_char_data_map(self, index: int, key: str, value: Union[bool, None, str, int, float]) -> bool:
        _data: dict[str, Union[bool, None, str, int, float]] = self._char_data_map.get(
            index, _MarkupTextManager.DEFAULT_CHAR_DATA.copy()
        )

        _previous: bool = _data.get(key, False)

        _data[key] = value
        self._char_data_map[index] = _data

        return _previous != value

    def _get_char_data_map(self, index: int, key: str, default: Any) -> Union[bool, None, str, int, float]:
        _data: dict[str, Union[bool, None, str, int, float]] = self._char_data_map.get(
            index, _MarkupTextManager.DEFAULT_CHAR_DATA.copy()
        )
        return _data.get(key, default)

    def _get_index_amount(self) -> int:
        return len(self._index_map.keys())

    def _add_char(self, index: int, char: str) -> None:
        self._lines[-1].append((index, char))
        self._char_map[index] = (len(self._lines) - 1, _get_string_unformatted(char))

    def get_unformatted_markup(self) -> str:
        return _get_string_unformatted(self._markup_text)

    def get_unformatted_text(self) -> str:
        return _get_string_unformatted(self._text)

    def _create_new_line(self) -> None:
        self._lines.append([])

    def set_text(self, text: str) -> None:
        self._reset(deep=True)
        self._set_text(text)

    def _set_text(self, text: str) -> None:
        self._text = text
        self._format()

    def _set_markup_text(self, text: str) -> None:
        self._markup_text = text
        self._label.text = text

    def get_cursor_x(self) -> float:
        return self._get_x(self._current_index)

    def get_cursor_y(self) -> float:
        target_index: int = self._current_index
        current_line: int = self._get_line_at(self._current_index)
        prev_index: int = max(0, self._current_index - 1)
        current_height: float = self._get_h(prev_index)
        adjust: float = 0

        if self._get_line_at(prev_index) == current_line:
            target_index = prev_index
            max_height: float = 0
            # find the biggest height in line
            if len(self._lines[current_line]) > 0:
                max_height: float = max(self._get_h(char[0]) for char in self._lines[current_line])

            if current_height != max_height:
                adjust = (max_height - current_height) / 2.0 + 1

        return self._get_y(target_index) - self._get_h(prev_index) - adjust

    def get_cursor_height(self) -> float:
        return self._get_h(max(0, self._current_index - 1))

    def update(self) -> None:
        self._refresh_callback(
            self.get_cursor_x(),
            self.get_cursor_y(),
            self.get_cursor_height(),
            self._selection_boxes
        )

    def update_deferred(self, timeout: int = 0) -> None:
        Clock.schedule_once(lambda _dt: self.update(), timeout)

    def _reset(self, deep: bool = False) -> None:
        self._lines = [[]]
        self._selection_boxes.clear()
        self._selected_indices.clear()
        self._link_indices.clear()
        self._initial_selected_index = 0
        self._char_map.clear()
        self._index_map.clear()

        if deep:
            self._char_data_map.clear()
            self._new_char_data_map.clear()

    def _format(self) -> None:
        self._adjust_indices()
        self._reset()

        _string_io: StringIO = StringIO()
        for index, char in enumerate(self._text):

            if char == "\n":
                _string_io.write(char)
                self._add_char(index, char)
                self._create_new_line()
            else:
                _string_io.write(f"{PRE_REF.format(index)}{char}{POST_REF}")
                self._add_char(index, char)

        self._markup_text = _string_io.getvalue()

        # apply char data
        for index, value in self._char_data_map.items():
            self._apply_char_data(index, value)
        self._label.text = self._markup_text
        self.update()

    def _apply_char_data(self, index, value):
        self._set_tag(index, value["bold"], PRE_BOLD, POST_BOLD)
        self._set_tag(index, value["italic"], PRE_ITALIC, POST_ITALIC)
        self._set_tag(index, value["underline"], PRE_UNDERLINE, POST_UNDERLINE)
        self._set_tag(index, value["strikethrough"], PRE_STRIKETHROUGH, POST_STRIKETHROUGH)
        self._set_tag(index, value["font"] is not None, PRE_FONT.format(value["font"]), POST_FONT)
        self._set_tag(index, value["font_size"] > 0, PRE_FONT_SIZE.format(value["font_size"]), POST_FONT_SIZE)

    def _get_current_line(self) -> int:
        return self._char_map.get(self._current_index, (len(self._lines) - 1, 0))[0]

    def _set_index(self, new_index: int) -> None:
        self._current_index = min(max(0, new_index), self._get_index_amount() - 1)
        self.update()

    def move_cursor_left(self) -> None:
        self._set_index(self._current_index - 1)

    def move_cursor_right(self) -> None:
        self._set_index(self._current_index + 1)

    def move_cursor_up(self) -> None:
        current_line: int = self._get_current_line()
        if current_line == 0:
            return

        _new_line: list = self._lines[current_line - 1]

        self._move_cursor_to_next_line(current_line, current_line - 1, _new_line)

    def move_cursor_down(self) -> None:
        current_line: int = self._get_current_line()
        if current_line == len(self._lines) - 1:
            return

        _new_line: list = self._lines[current_line + 1]

        self._move_cursor_to_next_line(current_line, current_line + 1, _new_line)

    def _move_cursor_to_next_line(self, current_line: int, next_line: int, new_line: list) -> None:

        _index_position: int
        if self._current_index == len(self._text):
            _index_position = len(self._lines[current_line])
        else:
            # get next best index
            _index_position = next(
                (
                    index
                    for index, item in enumerate(self._lines[current_line])
                    if item[0] == self._current_index
                ),
                0,
            )
        # if new line index is last line
        if next_line == len(self._lines) - 1 and _index_position >= len(new_line):  # adjust for fake last index
            # then set index to last char
            _index_position = len(new_line) - 1
            return self._set_index(new_line[_index_position][0] + 1)

        elif _index_position >= len(new_line):
            _index_position = len(new_line) - 1

        self._set_index(new_line[_index_position][0])

    def set_cursor_by_touch(self, touch: kivy.input.MotionEvent) -> None:
        self._set_index(self._get_closest_index(touch.x, touch.y))

    def _get_closest_index(self, x: float, y: float) -> int:
        x, y = self._label.to_local(x, y)
        closest_index: int = 0
        min_distance: float = math.inf
        target_point: tuple[float, float] = (x, y)
        for index in self._index_map.keys():
            other_point: tuple[float, float] = (self._get_x(index), self._get_y(index) - self._get_h(index) / 2.0)
            ecd: float = euclidean_distance(target_point, other_point)
            if ecd < min_distance:
                min_distance = ecd
                closest_index = index
        return closest_index

    def _rebuild_selection_boxes_deferred(self, timeout: int = 0) -> None:
        Clock.schedule_once(lambda _dt: self._rebuild_selection_boxes(), timeout)

    def _rebuild_selection_boxes(self) -> None:
        self._selection_boxes = []
        self._selected_indices.clear()
        selected_indices: list[int] = sorted(self._link_indices)

        # fill indices (if 1 and 10 is selected, select every number in between as well)
        if len(selected_indices) > 1:
            for index, value in enumerate(selected_indices.copy()):
                if index + 1 >= len(selected_indices):
                    break

                for filler in range(value + 1, list(selected_indices)[index + 1]):
                    if filler not in selected_indices:
                        selected_indices.append(filler)

        selected_indices = sorted(selected_indices)

        if not selected_indices:
            return

        # split indices in lines
        lines: list[list[int]] = [[] for _ in self._lines]
        for index in selected_indices:
            line_index: int = self._get_line_at(index)
            lines[line_index].append(index)

        # find max height in every line
        _max_height: float
        _line_height_map: list[float] = []

        for line in self._lines:
            _max_height = 0
            for char_index in line:
                _height: float = self._get_h(char_index[0])
                _max_height = max(_height, _max_height)

            _line_height_map.append(_max_height)

        for line_index, line in enumerate(lines):
            line = sorted(line)
            if not line:
                continue

            rectangle: list[float, float, float, float] = [0.0, 0.0, 0.0, 0.0]
            rectangle[0] = self._get_x(line[0])
            rectangle[1] = self._get_y(line[-1])
            rectangle[2] = self._get_x(line[-1]) - rectangle[0]
            rectangle[3] = _line_height_map[line_index]  # self._get_h(line[-1])
            self._selection_boxes.append(rectangle)

        # calculated actual selection by checking the selection direction
        backwards: bool = self._initial_selected_index > selected_indices[-1]

        if backwards:
            selected_indices.reverse()

        for index in selected_indices[:-1]:
            self._selected_indices.append(index)

    def start_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        _closest_index: int = self._get_closest_index(touch.x, touch.y)
        self._link_indices.clear()
        self._set_index(_closest_index)
        self._initial_selected_index = _closest_index

    def update_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        _closest_index: int = self._get_closest_index(touch.x, touch.y)
        self._link_indices = {self._initial_selected_index, _closest_index}
        self._rebuild_selection_boxes()
        self.update()

    def stop_select_by_drag(self, touch: kivy.input.MotionEvent) -> None:
        if len(self._link_indices) <= 1:
            self._link_indices.clear()
            self._rebuild_selection_boxes()
            self.update()

    def _adjust_indices(self) -> None:
        # swap indices in datamap
        # if newly inserted indices are {3, 4}
        # then the "old" 3,4 indices are now {5, 6} (and old 5, 6 is {7, 8} etc.)
        # now we need to shift every data index back
        # this should be relatively easy done by just increasing every index by the amount
        # of inserted indices after said indices
        if not self._new_char_data_map:
            return

        amount: int = len(self._new_char_data_map)
        start_index: int = sorted(list(self._new_char_data_map))[0]

        char_data_map_tmp: dict[int, dict[str, bool]] = {
            key + (amount if key >= start_index else 0): self._char_data_map[key]
            for key in self._char_data_map
        }
        # now add new indices
        char_data_map_tmp |= self._new_char_data_map

        # set new char data map
        self._char_data_map = char_data_map_tmp
        # clear map
        self._new_char_data_map.clear()

    def delete(self, left: bool = True) -> None:
        if self._selected_indices:
            current_index = self._selected_indices[0]
            self._set_text(self._text[:self._selected_indices[0]] + self._text[self._selected_indices[-1] + 1:])
            self._current_index = current_index
        else:
            delete_index: int = self._current_index
            if left and delete_index > 0:
                delete_index -= 1
            self._set_text(self._text[:delete_index] + self._text[delete_index + 1:])
            self._current_index = delete_index

        self.update_deferred()

    def insert(self, text: str, data: dict[str, Union[bool, None, str, int, float]] = None) -> None:
        # delete selected text
        if self._selected_indices:
            current_index = self._selected_indices[0]
            self._set_text(self._text[:self._selected_indices[0]] + self._text[self._selected_indices[-1] + 1:])
            self._current_index = current_index

        if data is None:
            data = _MarkupTextManager.DEFAULT_CHAR_DATA.copy()
        self._new_char_data_map.clear()

        # set new indices
        for c_index, char in enumerate(text):
            self._new_char_data_map[self._current_index + c_index] = data.copy()

        self._set_text(self._text[:self._current_index] + text + self._text[self._current_index:])
        self._current_index += len(text)

        self.update_deferred()

    def _find_ref_of_index(self, index: int, ignore: list = None) -> Union[tuple[int, int, str], None]:
        if ignore is None:
            ignore = []

        if self._get_char_at(index) in ignore:
            return logger.debug(f"Ignoring {index}")

        _query: str = PRE_REF.format(index)
        _start_index: int = self._markup_text.find(_query)
        if _start_index == -1:
            return logger.debug(f"Could not find index for {_query}")

        _start_index += len(_query)
        _end_index: int = self._markup_text.find(POST_REF, _start_index)
        _text: str = self._markup_text[_start_index:_end_index]

        return _start_index, _end_index, _text

    def _insert_at_index(self, index: int, pre: str = "", post: str = "", ignore: list[str] = None) -> None:
        # noinspection DuplicatedCode
        package: Union[tuple[int, int, str], None]

        if (package := self._find_ref_of_index(index, ignore)) is None:
            return

        _text: str
        _end_index: int
        _start_index: int
        _start_index, _end_index, _text = package

        self._set_markup_text(f"{self._markup_text[:_start_index]}{pre}{_text}{post}{self._markup_text[_end_index:]}")

    def _remove_at_index(self, index: int, pre: str = None, post: str = None, ignore: list[str] = None) -> None:
        # noinspection DuplicatedCode
        package: Union[tuple[int, int, str], None]

        if (package := self._find_ref_of_index(index, ignore)) is None:
            return

        _text: str
        _end_index: int
        _start_index: int
        _start_index, _end_index, _text = package

        # replace text
        if pre is not None:
            _text = _text.replace(pre, '')
        if post is not None:
            _text = _text.replace(post, '')

        # insert new text
        self._set_markup_text(f"{self._markup_text[:_start_index]}{_text}{self._markup_text[_end_index:]}")

    def _set_tag(self, index: int, value: bool, pre: str, post: str, ignore: list = None) -> None:
        if ignore is None:
            ignore = [' ']
        if value:
            self._insert_at_index(index, pre, post, ignore)
        else:
            self._remove_at_index(index, pre, post, ignore)

    def set_bold(self, value: bool) -> None:
        for index in self._selected_indices:
            if self._set_char_data_map(index, "bold", value):
                self._set_tag(index, value, PRE_BOLD, POST_BOLD)

    def set_italic(self, value: bool) -> None:
        for index in self._selected_indices:
            if self._set_char_data_map(index, "italic", value):
                self._set_tag(index, value, PRE_ITALIC, POST_ITALIC)

    def set_underline(self, value: bool) -> None:
        for index in self._selected_indices:
            if self._set_char_data_map(index, "underline", value):
                self._set_tag(index, value, PRE_UNDERLINE, POST_UNDERLINE, [])

    def set_strikethrough(self, value: bool) -> None:
        for index in self._selected_indices:
            if self._set_char_data_map(index, "strikethrough", value):
                self._set_tag(index, value, PRE_STRIKETHROUGH, POST_STRIKETHROUGH, [])

    def set_font(self, value: Union[str, None]) -> None:
        for index in self._selected_indices:
            _prev: int = self._get_char_data_map(index, "font", None)
            if self._set_char_data_map(index, "font", value):
                if _prev is not None:
                    self._remove_at_index(index, PRE_FONT.format(_prev), POST_FONT, [' '])
                self._set_tag(index, value is not None, PRE_FONT.format(value), POST_FONT)

        self._rebuild_selection_boxes_deferred()
        self.update_deferred()

    def set_font_size(self, value: Union[int, float]) -> None:
        for index in self._selected_indices:
            _prev: Union[float, int] = self._get_char_data_map(index, "font_size", -1)
            if self._set_char_data_map(index, "font_size", value):
                if _prev != -1:
                    self._remove_at_index(index, PRE_FONT_SIZE.format(_prev), POST_FONT_SIZE, [' '])
                self._set_tag(index, value > 0, PRE_FONT_SIZE.format(value), POST_FONT_SIZE)

        self._rebuild_selection_boxes_deferred()
        self.update_deferred()


class TextEdit(RelativeLayout):
    cursor_x = NumericProperty(0)
    cursor_y = NumericProperty(0)
    cursor_height = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._label = Label(
            text="-",
            halign="left",
            valign="top",
            markup=True,
            bold=True,
            color=(0, 0, 0),
            size_hint=(1, 1)
        )
        self.add_widget(self._label)
        self._gfx_selection: list = []

        self._manager: _MarkupTextManager = _MarkupTextManager(self._label, self._refresh_from_manager)

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

        self._setup()

    def on_size(self, w, v):
        self._label.text_size = v

    def set_text(self, text: str) -> None:
        self._manager.set_text(text)

    @mainthread
    def _refresh_from_manager(self, x: float, y: float, height: float, selection: list[list]) -> None:
        self.cursor_x = x
        self.cursor_y = y
        self.cursor_height = height

        for instruction in self._gfx_selection:
            self.canvas.remove(instruction)
        self._gfx_selection.clear()

        for box in selection:
            with self.canvas:
                c = Color(1, 1, 0, 0.5)
                r = Rectangle(pos=(box[0], box[1] - box[3]), size=(box[2], box[3]))
                self._gfx_selection.append(c)
                self._gfx_selection.append(r)

    def _setup(self):
        self.set_text("mmmmmmmmmm\nmmmmmmmm")

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


class Root(FloatLayout):
    def on_kv_post(self, base_widget):
        self.add_widget(TextEdit(size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.5}))


class TextEditApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        return Root()


if __name__ == "__main__":
    TextEditApp().run()
