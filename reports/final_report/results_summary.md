# Assignment 1A Results — Banking Regulatory Compliance

## Experiment configuration

- Domain: Enterprise Variant V3 — Banking / FinTech / Insurance regulatory compliance
- Model: HuggingFaceTB/SmolLM2-360M
- CPT split: 90% train / 10% held-out evaluation
- QLoRA adapters: A (r=8, alpha=16), B (r=16, alpha=32), C (r=32, alpha=32)
- Variant requirement: deterministic greedy decoding for compliance demonstrations

## Cleaning report

```csv
step,documents,removed,impact
input,127,0,baseline
input,127,0,
length_filter,125,2,
repetition_filter,125,0,
deduplication,123,2,
english_filter,118,5,greatest

```

## CPT loss summary

```json
{
  "initial_loss": 2.088490295410156,
  "final_loss": 2.1323215484619142,
  "loss_change_percent": 2.098705133947017,
  "plateau_step": 240
}
```

## Domain perplexity

```csv
base_model,cpt_model,evaluation_documents,base_ppl,cpt_ppl,percentage_improvement
HuggingFaceTB/SmolLM2-360M,/content/banking-compliance-llm-assignment-1a/models/cpt_model,12,8.352,7.7244,7.51

```

## Catastrophic forgetting comparison

| Prompt | Base model | CPT model | Verdict |
|---|---|---|---|
| The capital of France is | Paris. It is the largest city in France and the second largest in Europe. It is also the most populous city in Europe. It is the capital | the city of Paris. It is the largest city in France and the capital of the European Union. It is also the capital of the European country of | Retained |
| Water boils at | 100 degrees Celsius.  The boiling point of water is 100 degrees Celsius.  The boiling point of water is | 100 degrees Celsius.  The boiling point of water is 100 degrees Celsius.  The boiling point of water is | Retained |
| The speed of light is approximately | 186,000 miles per second.  The speed of light is the fastest speed in the universe.  The speed | 300,000 kilometers per second.  The speed of light is the fastest speed in the universe.  The speed | Retained |


## Adapter comparison

See `results/adapter_comparison/adapter_comparison.csv`. The table includes outputs for all three adapters and a transparent keyword-grounding proxy; review responses against the source corpus before making a final accuracy claim.

## Limitations

All metrics are data-dependent and must be regenerated after the student places the final public PDF corpus in `data/raw_pdfs/`. Regulatory answers require verification against current primary sources and are not legal or compliance advice.
