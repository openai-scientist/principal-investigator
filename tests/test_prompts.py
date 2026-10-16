"""Rules the prompts must state; live runs failed when one was missing."""


def test_the_prompts_say_where_checks_live_and_which_values_must_be_fixed():
    from principal_investigator.deliberation import prompt

    constitution = " ".join(prompt("constitution").split())
    assert "the hypothesis stage does not check datasets" in constitution
    assert "never send a stage back for a check that stage does not do" in constitution
    critic = " ".join(prompt("critic").split())
    assert "A rule that decides how results are analysed or read" in critic
