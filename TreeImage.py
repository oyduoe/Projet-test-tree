import math
from VectorImage import VectorImage

try:
    import svg as _svg
    _HAS_TITLE = hasattr(_svg, 'Title')
except Exception:
    _HAS_TITLE = False


class TreeImage(VectorImage):
    """Rend un arbre trinomial en SVG.

    Couleurs :
      - Fond du nœud = moneyness (S vs K).
          Call : vert foncé si deep ITM (S >> K), ambre si ATM, rouge si deep OTM.
          Put  : même logique en inversant.
      - Bordure épaisse rouge + point central blanc = exercice anticipé optimal
        (uniquement pour les options américaines).

    Taille du cercle ∝ sqrt(Cum_Proba).

    Arêtes colorées par direction :
      - Next_Up   : bleu
      - Next_Mid  : gris
      - Next_Down : orange
    """

    DEFAULT_WIDTH = 1600
    DEFAULT_HEIGHT = 1000

    def __init__(self, tree, max_depth, params=None, width=None, height=None):
        self.tree = tree
        self.max_depth = max(1, int(max_depth))
        self.params = params
        self.option_type = (getattr(params, 'type_contrat', 'Call') or 'Call') if params else 'Call'
        try:
            self.strike = float(getattr(params, 'strike', 0) or 0)
        except (TypeError, ValueError):
            self.strike = 0.0
        self.is_american = ((getattr(params, 'Exercice', 'European') or 'European')
                            == 'American') if params else False

        w = width or self.DEFAULT_WIDTH
        h = height or self.DEFAULT_HEIGHT
        super().__init__(w, h)
        self.build_image()

    # ------------------------------------------------------------------
    def build_image(self):
        columns, edges = self._tree_data()
        if not columns:
            return

        n_cols = len(columns)
        max_d = n_cols - 1

        left_margin, right_margin = 60, 60
        top_margin, bottom_margin = 40, 40
        usable_w = self.width - left_margin - right_margin
        usable_h = self.height - top_margin - bottom_margin

        col_width = (usable_w / max_d) if max_d > 0 else 0.0
        center_y = top_margin + usable_h / 2.0

        max_nodes = 2 * max_d + 1 if max_d > 0 else 1
        row_spacing = (usable_h / max_nodes) if max_nodes > 0 else usable_h
        if max_d > 0:
            max_radius = max(min(col_width * 0.35, row_spacing * 0.40), 3)
        else:
            max_radius = 20

        # --- Positions ---
        positions = {}
        id_to_node = {}
        for d, col in enumerate(columns):
            x = left_margin + d * col_width
            n = len(col)
            for idx, node in enumerate(col):
                k = idx - (n - 1) / 2.0
                y = center_y + k * row_spacing
                positions[id(node)] = (int(x), int(y))
                id_to_node[id(node)] = node

        # --- Arêtes (dessinées en premier, derrière) ---
        for (parent, attr, child) in edges:
            if id(parent) not in positions or id(child) not in positions:
                continue
            x1, y1 = positions[id(parent)]
            x2, y2 = positions[id(child)]
            if attr == "Next_Up":
                ec = '#5C9EE8'   # bleu
            elif attr == "Next_Down":
                ec = '#E89E5C'   # orange
            else:
                ec = '#B0B0B0'   # gris
            self.add_beginning(self.segment(x1, y1, x2, y2, ec, 1, opacity=0.35))

        # --- Nœuds ---
        for node_id, (x, y) in positions.items():
            node = id_to_node[node_id]
            radius = self._node_radius(node, max_radius)
            fill = self._node_fill(node)
            early_ex = self._is_early_exercise(node)

            if early_ex:
                stroke, sw = '#D32F2F', 3
            else:
                stroke, sw = '#333333', 1

            self.add_end(self.circle(x, y, radius, stroke, fill, sw, 0.95))

            # Point blanc central pour signaler l'exercice anticipé
            if early_ex and radius >= 5:
                inner = max(2, radius // 3)
                self.add_end(self.circle(x, y, inner,
                                         '#FFFFFF', '#FFFFFF', 0, 0.9))

    # ------------------------------------------------------------------
    def _tree_data(self):
        """Renvoie (columns, edges).

        - Si un snapshot post-pricing est disponible sur l'arbre, on l'utilise
          (les nœuds ont alors OptPrice rempli).
        - Sinon, BFS frais via Next_Up / Next_Mid / Next_Down.
        """
        snap = getattr(self.tree, '_tree_snapshot', None)
        if snap:
            return snap["columns"][:self.max_depth + 1], snap["edges"]

        root = self.tree.Root_Node
        if root is None:
            return [], []

        columns = [[root]]
        edges = []
        current = [root]
        seen = {id(root)}

        for _ in range(self.max_depth):
            nxt = []
            for node in current:
                for attr in ("Next_Up", "Next_Mid", "Next_Down"):
                    child = getattr(node, attr, None)
                    if child is None:
                        continue
                    edges.append((node, attr, child))
                    if id(child) not in seen:
                        nxt.append(child)
                        seen.add(id(child))
            if not nxt:
                break
            nxt.sort(key=lambda x: x.UndPrice, reverse=True)
            columns.append(nxt)
            current = nxt

        return columns, edges

    # ------------------------------------------------------------------
    def _node_radius(self, node, max_radius):
        p = getattr(node, 'Cum_Proba', 0) or 0.0
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 0.0
        p = max(0.0, min(1.0, p))
        r = max_radius * math.sqrt(max(p, 1e-4))
        return max(int(r), 2)

    def _is_early_exercise(self, node):
        if not self.is_american or self.params is None:
            return False
        opt = getattr(node, 'OptPrice', None)
        if opt is None:
            return False
        try:
            opt = float(opt)
        except (TypeError, ValueError):
            return False
        intrinsic = float(self.params.Vi(node.UndPrice))
        if intrinsic <= 0:
            return False
        tol = 1e-8 * max(1.0, abs(opt))
        return abs(opt - intrinsic) < tol

    def _node_fill(self, node):
        S = float(getattr(node, 'UndPrice', 0) or 0)
        if self.strike is None or self.strike <= 0 or S <= 0:
            return '#B0B0B0'

        if self.option_type == "Call":
            r = S / self.strike
        else:
            r = self.strike / S

        # r >= 1 → ITM → vert ; r < 1 → OTM → rouge
        # Interpolation sur [0.85, 1.15] : rouge → ambre → vert
        t = max(0.0, min(1.0, (r - 0.85) / 0.30))

        if t < 0.5:
            s = t / 0.5
            c1, c2 = (183, 28, 28), (255, 193, 7)      # rouge → ambre
        else:
            s = (t - 0.5) / 0.5
            c1, c2 = (255, 193, 7), (27, 94, 32)       # ambre → vert foncé

        rgb = tuple(int(c1[i] + (c2[i] - c1[i]) * s) for i in range(3))
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'