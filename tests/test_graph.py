from llm_soc_nav.graph import generic_adjacency_list, generic_graph_sentences, graph_sentences


def test_generic_graph_sentences_do_not_use_friendship_wording():
    names = list("abcdefghijklm")
    sentences = generic_graph_sentences(names)
    assert "a is connected to b." in sentences
    assert all("friends with" not in sentence for sentence in sentences)


def test_graph_sentences_selects_context():
    names = list("abcdefghijklm")
    assert "a is friends with b." in graph_sentences(names, "social")
    assert "a is connected to b." in graph_sentences(names, "generic")
    assert graph_sentences(names, "generic_adj_list")[0].startswith("Adjacency list:\nnode a: b, h")


def test_generic_adjacency_list_is_one_structured_block():
    names = list("abcdefghijklm")
    blocks = generic_adjacency_list(names)
    assert len(blocks) == 1
    assert "node b: a, c, d" in blocks[0]
    assert "is connected to" not in blocks[0]
