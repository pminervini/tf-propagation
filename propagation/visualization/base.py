# -*- coding: utf-8 -*-

from propagation.visualization.hinton import hinton_diagram


class HintonDiagram:
    def __call__(self, data):
        return hinton_diagram(data)

