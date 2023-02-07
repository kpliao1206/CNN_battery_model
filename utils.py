def shape_after_filter(IS, FS, padding, stride):
    OS = (IS - FS + 2*padding) / stride + 1
    return OS

# print(shape_after_filter(500, 5, 2, 1))
# print(shape_after_filter(500, 2, 0, 2))
# print(shape_after_filter(250, 2, 0, 2))
# print(shape_after_filter(250, 31, 0, 1))
# print(shape_after_filter(250, 11, 5, 1))
# print(shape_after_filter(250, 7, 3, 1))
# print(shape_after_filter(125, 7, 3, 1))
# print(shape_after_filter(125, 5, 2, 1))
print(shape_after_filter(500, 11, 5, 1))
print(shape_after_filter(500, 5, 2, 1))
print(shape_after_filter(500, 3, 1, 1))
print(shape_after_filter(125, 3, 1, 1))
print(shape_after_filter(125, 5, 2, 1))
print(shape_after_filter(125, 2, 0, 2))