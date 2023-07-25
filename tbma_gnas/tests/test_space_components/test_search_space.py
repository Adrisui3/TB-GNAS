from tbma_gnas.search_space import space_component
from tbma_gnas.search_space.component_type import ComponentType


class TestSpaceComponent:

    def test_non_negative_scores(self):
        layer = space_component.LearnableSpaceComponent(component_type=ComponentType.LAYER)
        for sc in layer.get_scores().values():
            assert sc == 1

    def test_learning_component(self):
        layer = space_component.LearnableSpaceComponent(component_type=ComponentType.LAYER)

        layer.learn("gatv1", 15)
        assert layer.get_scores()["gatv1"] == 16

        layer.learn("gatv1", -25)
        assert layer.get_scores()["gatv1"] == 1
