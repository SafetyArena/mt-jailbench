# Attacks

## How to create a new attack?

*Prerequisite: please read the [Developer Guide](./developer_guide.md) first.*

1. Open [attack_type.py](../engine/attack_type.py) and add a new Enum for your attack
2. Make a copy of the [template attack folder](../engine/attacks/_template/) and give it a name
3. Rename all Python files and class names in your new attack folder
4. Open `__init__.py` and change export names accordingly
5. Now you're ready to implement the details. Specifically,
    - `PromptGenerator` is responsible for generating and refining attack prompts. `next_prompt` function is invoked for the first attempt of each turn. In subsequent attempts of the same turn (i.e., when a turn is retried), `refine_prompt` function will be invoked instead. `system_prompt` function is used to set the system prompt for invoking the target model.
    - `OutcomeEvaluator` is responsible for judging the target model's response. Logic is implemented in `evaluate` function.
    - `FlowController` is responsible for controlling the attack trajectory. Logic is implemented in `next_action` function.
6. Finally, add your new attack to [jailbreak_engine.py](../engine/jailbreak_engine.py) to `_init_attack` function. This lets the framework use your new attack in both demo and benchmark modes.

## Available Attacks & Configs

### ActorBreaker (Actor)
- Paper: https://arxiv.org/pdf/2410.10700
- Original Repo: https://github.com/AI45Lab/ActorAttack

```py
{
    "actor_num": 3,
    "dynamic_retry": True,
    "try_multiple_actors": True,
    "attacker_model": {...},
    "judge_model": {...},
}
```
- `actor_num`: the number of actors to use. A full attack plan will be generated for each actor.
- `dynamic_retry`: if enabled, a prompt will be retried if the target model produces a 'rejective' response.
- `try_multiple_actors`: if enabled, the attack won't stop when one actor fails - it attempts to restart using a different actor.

### Crescendo
- Paper: https://crescendo-the-multiturn-jailbreak.github.io/assets/pdf/CrescendoFullPaper.pdf 
- Original Repo: https://github.com/Azure/PyRIT/tree/main

```py
{
    "max_refines_per_turn": 3,
    "attacker_model": {...},
    "judge_model": {...},
}
```
- `max_refines_per_turn`: maximum number of refinement steps allowed for a single turn.

### Chain-of-Attack (CoA)
- Paper: https://arxiv.org/pdf/2405.05610 
- Original Repo: https://github.com/YancyKahn/CoA

```py
{
    "n_init_chains": 3,
    "enable_attack_update": True,
    "max_update_retries": 3,
    "use_llm_for_similarity": False,
    "similarity_model": {...},
    "attacker_model": {...},
    "judge_model": {...},
}
```
- `n_init_chains`: number of initial attack chains to generate - will compare and proceed with the best one, others will be discarded
- `enable_attack_update`: whether to enabled prompt refinement (i.e. retry)
- `max_update_retries`: during refinement, COA requires updated query to be semantically close to the harmful target - otherwise it retries. This restricts how many time is can try
- `use_llm_for_similarity`: if true, will use an LLM to approximate similarity (only for quick testing, not recommended for experimentation)
- `similarity_model`: must specify `base_url` of similarity model; if using LLM approximation, must specify LLM details

### Foot-in-the-Door (FITD)
- Paper: https://arxiv.org/abs/2502.19820
- Original Repo: https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak

```py
{
    "change_prompt_retries": 3,
    "n_init_trajs": 3,
    "attacker_model": {...},
    "judge_model": {...},
}
```
- `change_prompt_retries`: the number of attempts when softening the initial prompt
- `n_init_trajs`: the number of initial trajectories to generate - will be merged together

### XTeaming
- Paper: https://arxiv.org/pdf/2504.13203
- Original Repo: https://github.com/salman-lui/x-teaming

```py
{
    "max_refines_per_turn": 3,
    "num_sets": 1,
    "use_multiple_strategies": True,
    "attacker_model": {...},
    "judge_model": {...},
    "textgrad_engine_model" {...}
}
```
- `textgrad_engine_model`: the engine model used by TextGrad library (can be same as attacker model)
- `max_refines_per_turn`: maximum number of refinement steps allowed for a single turn
- `num_sets`: each set will generate 10 attack strategies
- `use_multiple_strategies`: if one strategy fails, whether to restart the attack with the next strategy

### Mix
The mix attack allows arbitrarily combining components from the above 5 attacks.

```py
{
    "generator": AttackType.CRESCENDO,
    "updater": AttackType.X_TEAMING,
    "judge_and_flow": AttackType.X_TEAMING,
    "max_refines_per_turn": 3,
    "attacker_model": {...},
    "judge_model": {...},
    "use_llm_for_similarity": False,
    "similarity_model": {...},
}
```
- `generator`: the prompt generation module
- `updater`: the prompt refinement module
- `judge_and_flow`: the judge and control flow module (combined together)
- `max_refines_per_turn`: maximum number of refinement steps allowed for a single turn
- `use_llm_for_similarity` and `similarity_model` are needed for COA modules to work
