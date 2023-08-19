from tbma_gnas.search_space.search_space import SearchSpace


class TestSearchSpace:

    def test_input_output_blocks(self):
        space = SearchSpace(num_node_features=20, output_shape=5)
        _ = space.query_for_depth(depth=2)
        _ = space.query_for_depth(depth=3)

        for depth in space.space.keys():
            blocks = space.space[depth]
            assert blocks[0].get_input() and blocks[-1].get_output()

            if len(blocks) > 1:
                assert not blocks[-2].get_output()

    def test_num_blocks(self):
        space = SearchSpace(num_node_features=20, output_shape=5)
        model = space.query_for_depth(depth=2)
        assert len(model.get_blocks()) == 2

        model = space.query_for_depth(depth=3)
        assert len(model.get_blocks()) == 3

        model = space.query_for_depth(depth=4)
        assert len(model.get_blocks()) == 4

    def test_input_output_channels(self):
        space = SearchSpace(num_node_features=20, output_shape=5)
        model = space.query_for_depth(depth=2)

        blocks = model.get_blocks()
        assert blocks[0][0].in_channels == 20 and blocks[-1][0].out_channels == 5
