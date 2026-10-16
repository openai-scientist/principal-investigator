"""Rules the prompts must state; live runs failed when one was missing."""


def test_the_prompts_say_where_checks_live_and_which_values_must_be_fixed():
    from principal_investigator.deliberation import prompt

    constitution = " ".join(prompt("constitution").split())
    assert "the hypothesis stage does not check datasets" in constitution
    assert "never send a stage back for a check that stage does not do" in constitution
    critic = " ".join(prompt("critic").split())
    assert "A rule that decides how results are analysed or read" in critic


def test_a_selection_never_hands_the_next_stage_an_item_to_add():
    # Live 2026-10-16: the PI chose H1 alone and asked the plan stage to add a second hypothesis.
    from principal_investigator.deliberation import prompt

    text = " ".join(prompt("stage_outcome").split())
    assert "Never ask the next stage to add an item." in text
    assert "A rule that decides how results are analysed or read" in text
