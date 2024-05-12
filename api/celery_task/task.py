from celery import shared_task

from reaction_predicter import predict_proba_reaction

@shared_task(
        bind=True, 
        autoretry_for=(Exception,), retry_backoff=True, 
        retry_kwargs={"max_retries": 1},
        name='probability:get_probability_one'
)
def get_probability_task(
        request,
        reactive: list,
        product: str,
) -> float:
    proba = predict_proba_reaction.get_probability_one(reactive, product)
    return proba