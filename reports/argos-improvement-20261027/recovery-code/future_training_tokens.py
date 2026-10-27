"""Proposed future SFT composition. Does not change any frozen dataset or runtime."""

def compose(prefix_ids, completion_ids, *, max_length, loss_token_budget=None):
    prefix_ids=list(prefix_ids);completion_ids=list(completion_ids)
    if not prefix_ids or not completion_ids:raise ValueError('Empty prefix or completion')
    if type(max_length) is not int or max_length<=0:raise ValueError('Invalid sequence limit')
    if any(type(t) is not int or t<0 for t in prefix_ids+completion_ids):raise ValueError('Invalid token ID')
    if len(prefix_ids)+len(completion_ids)>max_length:raise ValueError('Explicitly handle overlength before training')
    budget=len(completion_ids) if loss_token_budget is None else loss_token_budget
    if type(budget) is not int or not 1<=budget<=len(completion_ids):raise ValueError('Invalid loss-token budget')
    ids=prefix_ids+completion_ids
    labels=[-100]*len(prefix_ids)+completion_ids[:budget]+[-100]*(len(completion_ids)-budget)
    assert ids[:len(prefix_ids)]==prefix_ids and len(ids)==len(labels)
    return dict(input_ids=ids,attention_mask=[1]*len(ids),labels=labels)
