import matplotlib as mpl

lightcyan, cyan = '#E0FFFF', '#00FFFF'  # 淡青,青色
lightcoral, red = '#F08080', '#FF0000'  # 淡珊瑚色,红色
linen, peachpuff = '#FAF0E6', '#FFDAB9'  # 亚麻色,桃红色
lightyellow, yellow = '#FFFFE0', '#FFFF00'  # 浅黄色,黄色
plum, violet = '#DDA0DD', '#EE82EE'  # 紫红色,紫罗兰色
lightblue, blue = '#ADD8E6', '#0000FF'  # 淡蓝色，蓝色
lightgreen, green = '#90EE90', '#008000'  # 淡绿色，绿色
lightpink, pink = '#FFB6C1', '#FFC0CB'  # 淡粉色，粉色


cm_light = {3: mpl.colors.ListedColormap([lightcyan, lightcoral, linen]),
            4: mpl.colors.ListedColormap([lightcyan, lightcoral, linen, lightyellow]),
            6: mpl.colors.ListedColormap([lightcyan, lightcoral, linen, lightyellow, plum, lightblue]),
            8: mpl.colors.ListedColormap([lightcyan, lightcoral, linen, lightyellow, plum, lightblue, lightgreen, lightpink])}

cm_dark = {3: mpl.colors.ListedColormap([cyan, red, peachpuff]),
           4: mpl.colors.ListedColormap([cyan, red, peachpuff, yellow]),
           6: mpl.colors.ListedColormap([cyan, red, peachpuff, yellow, violet, blue]),
           8: mpl.colors.ListedColormap([cyan, red, peachpuff, yellow, violet, blue, green, pink])}
