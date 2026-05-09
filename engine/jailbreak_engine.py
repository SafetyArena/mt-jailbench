import json
import logging
import os
import re

from client.unified_llm_client import UnifiedLLMClient

from .attack_type import AttackType
from .core import (
    ActionType,
    AttackAttempt,
    AttackContext,
    AttackOutcome,
    AttackTurn,
    OutcomeCode,
)
from .defense_type import DefenseType
from .interfaces.defense.text_processor import TextProcessor
from .utils.independent_judge import JailbreakJudge

logger = logging.getLogger(__name__)

# Some pretty formatting for Jailbreak Engine console logs
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
BLUE = "\033[96m"
RESET = "\033[0m"
_formatter = logging.Formatter(
    f"{BLUE}[Jailbreak Engine] %(message)s{RESET}"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class JailbreakEngine:
    def __init__(
        self,
        # below are required params for engine to work
        behavior_id: str,
        behavior: str,
        max_turns: int,
        max_epochs: int,
        independent_judge: bool,
        attack_type: AttackType,
        attack_config: dict,
        target_config: dict,
        # below are optional params
        defense_types: list[DefenseType] | None = None,
        defense_config: dict | None = None,
        independent_judge_config: dict | None = None,
    ) -> None:
        # initialize context - this is where all information will be stored
        self.context = AttackContext(
            harmful_behavior_id=behavior_id,
            harmful_behavior=behavior,
            max_turns=max_turns,
            max_epochs=max_epochs,
            history=[],
            full_history=[],
            current_turn=1,  # must start with 1
            current_epoch=0,  # must start with 0
            conversation=None,
            attack_outcome=None
        )

        # initialize independent judge
        self.enable_inpd_judge = independent_judge
        self.inpd_judge_result = {"status": "disabled"}
        self.judge_types = []
        self.judge_threshold = 0
        if independent_judge:
            if not independent_judge_config:
                raise ValueError("Must provide independent_judge_config to enable independent judge")
            self.inpd_judge_result["status"] = "enabled but not triggered"
            self.judge = JailbreakJudge(independent_judge_config)
            judge_types_raw = independent_judge_config["judge_types"]
            self.judge_types = [JailbreakJudge.Type(j) for j in judge_types_raw]
            self.judge_threshold = independent_judge_config["majority_vote_threshold"]

        # initialize target
        self.target = UnifiedLLMClient(
            model=target_config["model"],
            provider=target_config.get("provider"),
            base_url=target_config.get("base_url")
        )

        # initialize attack (can only have one) and defenses (can have multiple)
        self._init_attack(attack_type, attack_config, target_config)
        self._init_defenses(defense_types, defense_config, target_config)
    
    def _init_attack(self, attack_type, attack_config, target_config):
        logger.info(f"Initializing {attack_type.value} attack...")
        attack_config["mirror_target_config"] = target_config.copy()

        match attack_type:
            case AttackType.ACTOR:
                from .attacks.actor import (
                    ActorFlowController,
                    ActorOutcomeEvaluator,
                    ActorPromptGenerator,
                )
                self.prompt_generator = ActorPromptGenerator(self.context, attack_config)
                self.outcome_evaluator = ActorOutcomeEvaluator(self.context, attack_config)
                self.flow_controller = ActorFlowController(self.context, attack_config)

            case AttackType.COA:
                from .attacks.coa import (
                    CoAFlowController,
                    CoAOutcomeEvaluator,
                    CoAPromptGenerator,
                )
                self.prompt_generator = CoAPromptGenerator(self.context, attack_config)
                self.outcome_evaluator = CoAOutcomeEvaluator(self.context, attack_config)
                self.flow_controller = CoAFlowController(self.context, attack_config)

            case AttackType.CRESCENDO:
                from .attacks.crescendo import (
                    CrescendoFlowController,
                    CrescendoOutcomeEvaluator,
                    CrescendoPromptGenerator,
                )
                self.prompt_generator = CrescendoPromptGenerator(self.context, attack_config)
                self.outcome_evaluator = CrescendoOutcomeEvaluator(self.context, attack_config)
                self.flow_controller = CrescendoFlowController(self.context, attack_config)

            case AttackType.FITD:
                from .attacks.fitd import (
                    FitdFlowController,
                    FitdOutcomeEvaluator,
                    FitdPromptGenerator,
                )
                self.prompt_generator = FitdPromptGenerator(self.context, attack_config)
                self.outcome_evaluator = FitdOutcomeEvaluator(self.context, attack_config)
                self.flow_controller = FitdFlowController(self.context, attack_config)

            case AttackType.X_TEAMING:
                from .attacks.xteaming import (
                    XTeamingFlowController,
                    XTeamingOutcomeEvaluator,
                    XTeamingPromptGenerator,
                )
                self.prompt_generator = XTeamingPromptGenerator(self.context, attack_config)
                self.outcome_evaluator = XTeamingOutcomeEvaluator(self.context, attack_config)
                self.flow_controller = XTeamingFlowController(self.context, attack_config)

            case AttackType.INTERACTIVE:
                from .attacks.interactive import (
                    InteractiveFlowController,
                    InteractiveOutcomeEvaluator,
                    InteractivePromptGenerator,
                )
                self.prompt_generator = InteractivePromptGenerator()
                self.outcome_evaluator = InteractiveOutcomeEvaluator()
                self.flow_controller = InteractiveFlowController()

            case AttackType.MIX:
                from .attacks.mix import (
                    MixFlowController,
                    MixOutcomeEvaluator,
                    MixPromptGenerator,
                )
                self.prompt_generator = MixPromptGenerator(self.context, attack_config)
                self.outcome_evaluator = MixOutcomeEvaluator(self.context, attack_config)
                self.flow_controller = MixFlowController(self.context, attack_config)

    def _init_defenses(self, defense_types, defense_config, target_config):
        # list of tuples (text_processor, defense_name)
        self.defenses: list[tuple[TextProcessor, str]] = []

        if not defense_types:
            return

        # initialize defenses in the same order as user specified
        for defense_type in defense_types:
            logger.info(f"Initializing {defense_type.value} defense...")

            # defense config looks something like { defense_name: {...} }
            # so extract the specific config for each defense
            specific_config = defense_config.get(defense_type.value, {})
            specific_config["mirror_target_config"] = target_config.copy()

            text_processor = None
            match defense_type:
                case DefenseType.DUMMY:
                    from .defenses.dummy import DummyTextProcessor
                    text_processor = DummyTextProcessor(self.context, specific_config)
                case _:
                    raise ValueError(f"Defense type {defense_type} not supported")

            self.defenses.append((text_processor, defense_type.value))

    def execute(self) -> None:
        """
        Execute a single harmful behavior.
        """
        try:
            context = self.context
            logger.info(f"Harmful behavior: {context.harmful_behavior}")

            while True:
                # terminate if max_turns is reached
                if context.current_turn > context.max_turns:
                    self._terminate(OutcomeCode.MAX_TURN_REACHED)

                # create a new turn and execute it
                turn = AttackTurn(context.current_turn)
                context.history.append(turn)
                self._execute_turn(turn)

                # use next_action from the last attempt of this turn
                next_action = turn.all_attempts[-1].next_action
                match next_action.type:
                    case ActionType.RETRY | ActionType.END_SUCCESS | ActionType.END_FAILURE:
                        raise RuntimeError(f"Internal error: {next_action.type} shouldn't be handled by execute()")
                    case ActionType.CONTINUE:
                        context.current_turn += 1
                    case ActionType.JUMP_TO:
                        # jump_to turn must be smaller than current turn
                        if next_action.payload > context.current_turn:
                            raise ValueError("JUMP_TO turn must be smaller or equal to current turn")
                        # truncate history and conversation
                        context.current_turn = next_action.payload
                        context.history = context.history[:next_action.payload - 1]
                        if context.current_turn == 1:
                            context.conversation = None
                        else:
                            last_turn = context.get_turn(next_action.payload - 1)
                            context.conversation = last_turn.attempt_in_effect._conv_after_response
        except TerminateExecution:
            # benign exception, used for terminating execution
            pass

    def _execute_turn(self, turn: AttackTurn) -> None:
        """
        Run a single turn.
        """
        context = self.context

        while True:
            context.current_epoch += 1

            # terminate if max_epochs is reached
            if context.current_epoch > context.max_epochs:
                # special case: runs out of epoch and leaves an empty turn
                if turn.attempt_in_effect is None:
                    context.history.pop()
                self._terminate(OutcomeCode.MAX_EPOCH_REACHED)

            attempt = AttackAttempt(epoch=context.current_epoch)

            turn.all_attempts.append(attempt)
            turn.attempt_in_effect = attempt  # default attempt_in_effect to the latest attempt
            context.full_history.append(attempt)

            self._execute_attempt(attempt)

            match attempt.next_action.type:

                case ActionType.END_SUCCESS:
                    outcome_code = OutcomeCode.SUCCESS_BY_SELF_EVAL
                    if self.enable_inpd_judge:
                        outcome_code = (
                            OutcomeCode.SUCCESS_BY_INDP_EVAL
                            if self._independent_judging()
                            else OutcomeCode.FAILURE_BY_INDP_EVAL
                        )
                    self._terminate(outcome_code)

                case ActionType.END_FAILURE:
                    self._terminate(OutcomeCode.FAILURE_BY_SELF_EVAL)

                case ActionType.CONTINUE:
                    # logic below handles optional parameter that allows continue with specified attempt
                    preferred_attempt = attempt.next_action.payload
                    if preferred_attempt is not None:
                        if not isinstance(preferred_attempt, int):
                            raise ValueError("CONTINUE payload must be epoch number")

                        matching_attempt = next((a for a in turn.all_attempts if a.epoch == preferred_attempt), None)
                        if matching_attempt is None:
                            raise ValueError(f"No attempt found with epoch {preferred_attempt} in this turn")
                        turn.attempt_in_effect = matching_attempt
                        context.conversation = matching_attempt._conv_after_response
                    break

                case ActionType.RETRY:
                    # reset conversation
                    if context.current_turn == 1:
                        context.conversation = None
                    else:
                        last_turn = context.get_turn(context.current_turn - 1)
                        context.conversation = last_turn.attempt_in_effect._conv_after_response
                    pass

                case ActionType.JUMP_TO:
                    break

    def _execute_attempt(self, attempt: AttackAttempt) -> None:
        """
        Run a single attempt.
        """
        context = self.context
        logger.info(f"Turn {context.current_turn} (epoch {attempt.epoch}):")

        # step 1: get prompt from PromptGenerator
        attempt.system_prompt = self.prompt_generator.system_prompt(context)
        if len(context.history[-1].all_attempts) > 1:
            attempt.prompt = self.prompt_generator.refine_prompt(context)
        else:
            attempt.prompt = self.prompt_generator.next_prompt(context)
        self._pretty_log("Prmp", attempt.prompt)

        # step 1.5: if defense(s) is configured, pre-process the prompt
        if self.defenses:
            attempt.pre_trace = [{"stage": "raw", "text": attempt.prompt}]
        for tp, name in self.defenses:
            result = tp.process_prompt(context)
            if result is None:
                continue  # None means no change
            attempt.prompt = result
            attempt.pre_trace.append({"stage": f"after {name}", "text": attempt.prompt})
            self._pretty_log("Prmp", attempt.prompt, True)

        # step 2: send prompt to target model and get response
        attempt.response, attempt._conv_after_response = self.target.generate(
            user_input=attempt.prompt,
            conversation=context.conversation,
            system_prompt=attempt.system_prompt,
        )
        if attempt.response is None:
            logger.warning("Target model returned None (likely due to content filter), using dummy response instead")
            attempt.response = "I’m sorry, I can’t help with that request."
        self._pretty_log("Resp", attempt.response)
        context.conversation = attempt._conv_after_response

        # step 2.5: if defense(s) is configured, post-process the response
        if self.defenses:
            attempt.post_trace = [{"stage": "raw", "text": attempt.response}]
        for tp, name in self.defenses:
            result = tp.process_response(context)
            if result is None:
                continue  # None means no change
            attempt.response = result
            attempt.post_trace.append({"stage": f"after {name}", "text": attempt.response})
            self._pretty_log("Resp", attempt.response, True)

        # step 3: send context to OutcomeEvaluator and get evaluation
        attempt.evaluation = self.outcome_evaluator.evaluate(context)
        self._pretty_log("Eval", str(attempt.evaluation))

        # step 4: send context to State Updater and get next action
        attempt.next_action = self.flow_controller.next_action(context)
        self._pretty_log("Actn", str(attempt.next_action))
    
    def _pretty_log(self, prefix: str, text: str, mark: bool=False):
        logger.info(
            " |%s%s: %s",
            "*" if mark else " ",
            prefix,
            text[:80].replace("\n", "\\n") + ("... (truncated)" if len(text) > 80 else "")
        )

    def _terminate(self, outcome_code: OutcomeCode):
        """
        Terminate the engine execution at any time.
        """
        logger.info(f"Attack terminated with signal {outcome_code.name}")
        match outcome_code:
            case OutcomeCode.SUCCESS_BY_SELF_EVAL | OutcomeCode.SUCCESS_BY_INDP_EVAL:
                self.context.attack_outcome = AttackOutcome(True, outcome_code)
            case _:
                self.context.attack_outcome = AttackOutcome(False, outcome_code)
        raise TerminateExecution()

    def _independent_judging(self) -> bool:
        """
        Only invoked if:
         - independent judge is enabled, and
         - attack is successful in self-reported evaluation

        Return whether the jailbreak was successful based on independent judge.
        """
        final_model_rsps = self.context.conversation.history[-1]["content"]
        is_successful, details = self.judge.majority_vote(
            behavior=self.context.harmful_behavior,
            response=final_model_rsps,
            types=self.judge_types,
            threshold=self.judge_threshold
        )

        self.inpd_judge_result["status"] = "triggered"
        self.inpd_judge_result["attack_successful"] = is_successful
        self.inpd_judge_result["details"] = details

        logger.info("Independent judge:")
        logger.info(f" | Jailbreak {"SUCCEEDED" if is_successful else "FAILED"}")
        return is_successful

    def retrieve_result(self):
        """
        Return the result of engine execution as a dictionary.
        """
        context = self.context
        return {
            "behavior_id": context.harmful_behavior_id,
            "behavior": context.harmful_behavior,
            "max_turns": context.max_turns,
            "max_epochs": context.max_epochs,
            "total_turns_used": len(context.history),
            "total_epochs_used": len(context.full_history),
            "attack_successful": context.attack_outcome.successful,
            "outcome_code": context.attack_outcome.outcome_code.name,
            "independent_judge": self.inpd_judge_result,
        }

    def log_result(self, output_path: str) -> None:
        """
        Log the result of engine execution as JSON file (contains more details than retrieve_result).
        """
        context = self.context
        log = {
            "behavior_id": context.harmful_behavior_id,
            "behavior": context.harmful_behavior,
            "max_turns": context.max_turns,
            "max_epochs": context.max_epochs,
            "total_turns_used": len(context.history),
            "total_epochs_used": len(context.full_history),
            "attack_successful": context.attack_outcome.successful,
            "outcome_code": context.attack_outcome.outcome_code.name,
            "independent_judge": self.inpd_judge_result,
            "final_conversation": context.conversation.serialize() if context.conversation else [],
            "history":  [t.to_dict() for t in context.history],
            "full_history": [t.to_dict() for t in context.full_history],
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            s = json.dumps(log, ensure_ascii=False, indent=2)
            # collapse defense tracing dicts like {"stage": "...", "text": "..."}
            s = re.sub(
                r'\{\n\s+"stage": "(.*?)",\n\s+"text": "(.*?)"\n\s+\}',
                r'{"stage": "\1", "text": "\2"}',
                s
            )
            # collapse evaluation dict
            s = re.sub(
                r'"evaluation":\s*\{\n(.*?)\n\s*\}',
                lambda m: '"evaluation": {' +
                        re.sub(r'\s+', ' ', m.group(1)).strip() +
                        '}',
                s,
                flags=re.DOTALL
            )
            f.write(s)
        logger.info(f"Full audit log saved to {output_path}")


class TerminateExecution(Exception):
    """Raised to stop the whole execution early"""
    pass
