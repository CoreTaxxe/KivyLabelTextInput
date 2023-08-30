from kivy.core.text import Label
from kivy.core.text.text_layout import layout_text

l = Label(font_size=20, bold=True)
lines = []
print(l.options)
# layout text with width constraint by 50, but no height constraint
w, h, clipped = layout_text('test', lines, (0, 0), (50, None), l.options,
                            l.get_cached_extents(), True, False)
print(w, h, clipped)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
print(lines)
w, h, clipped = layout_text(' ', lines, (w, h), (50, None), l.options, l.get_cached_extents(), True, False)
print(lines)
for li in lines:
    for wo in li.words:
        print(wo.text, end=" ")
    print()
#print(w, h, clipped)
