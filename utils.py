# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 17:39 2023/8/26
import math


def strict_round(number):
    """
    严格的四舍五入函数(对于只有一位小数的数字来说)
    因为python自带的round不是严格的四舍五入(2.5->2  3.5->4)
    用Decimal会更好，这里只是自己写一下
    :param number: 输入的需要四舍五入的数字（只针对只有一位小数的数字）
    :return: 整型数字
    """
    int_part = math.floor(number)
    ten_int_part = math.floor(number * 10)
    if ten_int_part % 5 != 0:
        return round(number)
    if (ten_int_part / 5) % 2 == 0:
        return round(number)
    if int_part % 2 != 0:
        return round(number)
    return round(number) + 1