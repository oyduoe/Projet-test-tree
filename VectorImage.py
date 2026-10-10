import svg


class VectorImage:
    """Interface entre le package svg et le reste du code.

    Cette classe n'a aucune dépendance vers Tree / Monte Carlo / etc. :
    si on change de package de rendu, on ne modifie que ce fichier.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.image = svg.SVG(width=width, height=height, elements=[])

    # ------------------------------------------------------------------
    # Gestion des éléments
    # ------------------------------------------------------------------
    def add_end(self, element):
        """Ajoute un élément à la fin (affiché au-dessus des précédents)."""
        self.image.elements += [element]

    def add_beginning(self, element):
        """Insère un élément au début (affiché en dessous)."""
        self.image.elements.insert(0, element)

    # ------------------------------------------------------------------
    # Fabriques d'éléments géométriques
    # ------------------------------------------------------------------
    @staticmethod
    def circle(center_x, center_y, radius,
               stroke_color, fill_color, stroke_width, opacity=1.0):
        return svg.Circle(
            cx=center_x, cy=center_y, r=radius,
            stroke=stroke_color, fill=fill_color,
            stroke_width=stroke_width, opacity=opacity)

    @staticmethod
    def rectangle(top_left_x, top_left_y, width, height,
                  stroke_color, fill_color, stroke_width,
                  rounded_corner=0, opacity=1.0):
        return svg.Rect(
            x=top_left_x, y=top_left_y, width=width, height=height,
            rx=rounded_corner,
            stroke=stroke_color, fill=fill_color,
            stroke_width=stroke_width, opacity=opacity)

    @staticmethod
    def segment(p1_x, p1_y, p2_x, p2_y,
                stroke_color, stroke_width, opacity=1.0):
        return svg.Line(
            x1=p1_x, y1=p1_y, x2=p2_x, y2=p2_y,
            stroke=stroke_color, stroke_width=stroke_width, opacity=opacity)

    # ------------------------------------------------------------------
    # Sortie
    # ------------------------------------------------------------------
    def as_str(self) -> str:
        return self.image.as_str()

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(self.image.as_str())