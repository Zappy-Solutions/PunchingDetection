# utils.py
def point_line_distance(point, line):
    """
    Calculate the perpendicular distance from a point to a line.
    """
    (x0, y0) = point
    ((x1, y1), (x2, y2)) = line
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return numerator / denominator if denominator != 0 else float('inf')