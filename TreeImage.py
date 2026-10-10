import math
from VectorImage import VectorImage


class TreeImage(VectorImage):
    """Rend un arbre trinomial en SVG.

    Points clés :
    - On ne parcourt que `max_depth` colonnes (les premières de l'arbre).
      Même pour un arbre à 10 000 pas, la construction reste très rapide.
    - Colonne 0 = uniquement la racine (les UpNode/DownNode éventuellement
      attachés à la racine ne font pas partie de l'arbre : ils servent au
      calcul des Greeks par différences finies sur les 3 nœuds racine).
    - Couleur d'un nœud : dégradé vert selon la proba cumulée, rouge si
      l'une des probabilités de transition est négative.
    - Rayon ∝ sqrt(proba cumulée).
    - Les arêtes sont dessinées en premier (add_beginning) pour rester
      derrière les nœuds.
    """

    DEFAULT_WIDTH = 1600
    DEFAULT_HEIGHT = 1000

    def __init__(self, tree, max_depth, width=None, height=None):
        self.tree = tree
        self.max_depth = max(1, int(max_depth))
        w = width or self.DEFAULT_WIDTH
        h = height or self.DEFAULT_HEIGHT
        super().__init__(w, h)
        self.build_image()

    # ------------------------------------------------------------------
    def build_image(self):
        columns = self._collect_columns(self.max_depth)
        if not columns:
            return

        # Cas dégénéré : seulement la racine
        if len(columns) == 1:
            cx, cy = self.width // 2, self.height // 2
            self.add_end(self.circle(cx, cy, 20, '#222222', '#2A9D8F', 1, 1.0))
            return

        n_cols = len(columns)
        max_d = n_cols - 1

        left_margin = 60
        right_margin = 60
        top_margin = 40
        bottom_margin = 40

        usable_w = self.width - left_margin - right_margin
        usable_h = self.height - top_margin - bottom_margin

        col_width = usable_w / max_d
        center_y = top_margin + usable_h / 2.0

        max_nodes = 2 * max_d + 1
        row_spacing = usable_h / max_nodes
        max_radius = max(min(col_width * 0.35, row_spacing * 0.40), 2)

        # --- Positions & index ---
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

        # --- Arêtes (derrière) ---
        for d in range(n_cols - 1):
            next_col_ids = {id(n) for n in columns[d + 1]}
            for node in columns[d]:
                x1, y1 = positions[id(node)]
                for attr in ("Next_Up", "Next_Mid", "Next_Down"):
                    child = getattr(node, attr, None)
                    if child is None or id(child) not in next_col_ids:
                        continue
                    x2, y2 = positions[id(child)]
                    edge_color = ('#F4A4A4' if self._has_negative(node)
                                  else '#B8B8B8')
                    self.add_beginning(
                        self.segment(x1, y1, x2, y2, edge_color, 1,
                                     opacity=0.35))

        # --- Nœuds ---
        for node_id, (x, y) in positions.items():
            node = id_to_node[node_id]
            radius = self._node_radius(node, max_radius)
            fill = self._node_color(node)
            self.add_end(self.circle(
                x, y, radius,
                stroke_color='#333333', fill_color=fill,
                stroke_width=1, opacity=0.95))

    # ------------------------------------------------------------------
    def _collect_columns(self, max_depth):
        """Parcourt l'arbre colonne par colonne.

        - Colonne 0 : uniquement la racine (pas de chaîne Up/Down : les
          UpNode/DownNode de la racine ne sont pas des nœuds de l'arbre,
          ce sont des artefacts pour le calcul des Greeks).
        - Colonnes suivantes : chaîne des UpNode au-dessus du mid,
          chaîne des DownNode en-dessous du mid.
        """
        columns = []
        root = self.tree.Root_Node
        if root is None:
            return columns

        # --- Colonne 0 : la racine seule ---
        columns.append([root])

        # --- Colonnes suivantes : à partir du mid du pas suivant ---
        current_node = root.Next_Mid

        for _ in range(max_depth):
            if current_node is None:
                break

            col = []
            mid_node = current_node

            # chaîne des "Up" depuis le mid (inclut le mid)
            n = mid_node
            while n is not None:
                col.append(n)
                n = n.UpNode

            # chaîne des "Down" sous le mid (mid déjà ajouté)
            n = mid_node.DownNode
            while n is not None:
                col.append(n)
                n = n.DownNode

            # Tri haut -> bas par prix décroissant
            col.sort(key=lambda x: x.UndPrice, reverse=True)
            columns.append(col)

            current_node = mid_node.Next_Mid

        return columns

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

    def _has_negative(self, node):
        for attr in ('Proba_Up', 'Proba_Mid', 'Proba_Down'):
            v = getattr(node, attr, 0) or 0
            try:
                if float(v) < 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _node_color(self, node):
        if self._has_negative(node):
            return '#E63946'  # rouge
        p = getattr(node, 'Cum_Proba', 0) or 0.0
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 0.0
        p = max(0.0, min(1.0, p))
        # Dégradé vert clair -> vert foncé
        r = int(200 - 160 * p)
        g = int(220 - 60 * p)
        b = int(200 - 160 * p)
        return f'#{r:02x}{g:02x}{b:02x}'