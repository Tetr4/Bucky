from typing import NamedTuple


ColorBGR = NamedTuple("ColorBGR", [('b', int), ('g', int), ('r', int)])
Point2D = NamedTuple("Point2D", [('x', float), ('y', float)])
Point3D = NamedTuple("Point3D", [('x', float), ('y', float), ('z', float)])
EulerAngles = NamedTuple("EulerAngles", [('pitch', float), ('yaw', float), ('roll', float)])
