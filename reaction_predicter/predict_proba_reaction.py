import time

def get_probability_one(
        reactive: list,
        product: str
) -> float:
    
    time.sleep(30)
    return {"reaction": f"{'+'.join(reactive)} = {product}", 'proba': 0.6}