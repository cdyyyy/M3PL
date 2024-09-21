from dassl.evaluation.evaluator import *
import torch.nn.functional as F
from dassl.utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry("EVALUATOR")

def build_multi_evaluator(cfg, **kwargs):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.EVALUATOR, avai_evaluators)
    method = cfg.TEST.METHOD
    if cfg.VERBOSE:
        print("Loading evaluator: {}".format(cfg.TEST.EVALUATOR))
    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg, method=method, **kwargs)

@EVALUATOR_REGISTRY.register()
class MultiClassification(EvaluatorBase):
    """Evaluator for classification with multiple prompts."""

    def __init__(self, cfg, lab2cname=None, method='avg', **kwargs):
        super().__init__(cfg)
        print(f"evaluator method: {method}")
        self._lab2cname = lab2cname
        self._correct_avg = 0
        if cfg.TRAINER.NAME == 'IVLP':
            print("=> using IVLP")
            self.n_prompts = cfg.TRAINER.IVLP.N_PROMPTS
        elif cfg.TRAINER.NAME == 'M3PL':
            print("=> using M3PL")
            self.n_prompts = cfg.TRAINER.M3PL.N_PROMPTS
        if method == 'all':
            self._correct_min_entropy = 0
            self._correct_max_confidence = 0
            self._correct_ind = [0] * self.n_prompts
        self._total = 0
        self._per_class_res_avg = None
        self.method = method
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res_avg = defaultdict(list)
            if method == 'all':
                self._per_class_res_min_entropy = defaultdict(list)
                self._per_class_res_max_confidence = defaultdict(list)

    def reset(self):
        self._correct_avg = 0
        if self.method == 'all':
            self._correct_min_entropy = 0
            self._correct_max_confidence = 0
            self._correct_ind = [0] * self.n_prompts
        self._total = 0
        if self._per_class_res_avg is not None:
            self._per_class_res_avg = defaultdict(list)
            if self.method == 'all':
                self._per_class_res_min_entropy = defaultdict(list)
                self._per_class_res_max_confidence = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): if method == avg, model output [batch, num_classes]; else model output [batch, n_prompts, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        if self.method == 'avg':
            logit_avg = mo.mean(dim=0)
            pred = logit_avg.max(1)[1]
            matches = pred.eq(gt).float()
            self._correct_avg += int(matches.sum().item())
            self._total += gt.shape[0]

            if self._per_class_res is not None:
                for i, label in enumerate(gt):
                    label = label.item()
                    matches_i = int(matches[i].item())
                    self._per_class_res_avg[label].append(matches_i)
        elif self.method == 'one':
            logits = mo
            pred = logits.max(1)[1]
            matches = pred.eq(gt).float()
            self._correct_avg += int(matches.sum().item())
            self._total += gt.shape[0]
        else:
            mo = mo.permute(1, 0, 2)
            self._total += gt.shape[0]
            logit_avg = mo.mean(dim=0)
            pred_avg = logit_avg.max(1)[1]
            matches_avg = pred_avg.eq(gt).float()
            self._correct_avg += int(matches_avg.sum().item())
            print(f'num prompts: {self.n_prompts}')
            for i in range(self.n_prompts):
                logit_i = mo[i]
                pred_i = logit_i.max(1)[1]
                matches_i = pred_i.eq(gt).float()
                self._correct_ind[i] += int(matches_i.sum().item())

            probs = F.softmax(mo, dim=-1)
            entropies = -torch.sum(probs * torch.log(probs), dim=-1)
            min_entropy_idx = torch.argmin(entropies, dim=0)
            logit_min_entropy = mo[min_entropy_idx, torch.arange(mo.shape[1])]
            pred_min_entropy = logit_min_entropy.max(1)[1]
            matches_min_entropy = pred_min_entropy.eq(gt).float()
            self._correct_min_entropy += int(matches_min_entropy.sum().item())
            
            max_calss_probs, _ = torch.max(probs, dim=-1)
            max_confidence_idx = torch.argmax(max_calss_probs, dim=0)
            logit_max_confidence = mo[max_confidence_idx, torch.arange(mo.shape[1])]
            pred_max_confidence = logit_max_confidence.max(1)[1]
            matches_max_confidence = pred_max_confidence.eq(gt).float()
            self._correct_max_confidence += int(matches_max_confidence.sum().item())

            if self._per_class_res_avg is not None:
                for i, label in enumerate(gt):
                    label = label.item()
                    matches_i_avg = int(matches_avg[i].item())
                    matches_i_min_entropy = int(matches_min_entropy[i].item())
                    matches_i_max_confidence = int(matches_max_confidence[i].item())
                    self._per_class_res_avg[label].append(matches_i_avg)
                    self._per_class_res_min_entropy[label].append(matches_i_min_entropy)
                    self._per_class_res_max_confidence[label].append(matches_i_max_confidence)
    
    def evaluate(self):
        results = OrderedDict()
        acc_avg = 100.0 * self._correct_avg / self._total
        if self.method == 'all':
            acc_min_entropy = 100.0 * self._correct_min_entropy / self._total
            acc_max_confidence = 100.0 * self._correct_max_confidence / self._total
            acc_ind = [0.0] * self.n_prompts
            for i in range(len(self._correct_ind)):
                acc_ind[i] = 100.0 * self._correct_ind[i] / self._total
        
        results['acc_avg'] = acc_avg
        if self.method == 'all':
            results['acc_min_entropy'] = acc_min_entropy
            results['acc_max_confidence'] = acc_max_confidence
            for i in range(len(self._correct_ind)):
                results[f'acc_prompt_{i}'] = acc_ind[i]

        if self.method == 'avg':
            print(
                "=> result\n"
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct_avg:,}\n"
                f"* accuracy: {acc_avg:.2f}%\n"
            )
        elif self.method == 'one':
            print(
                "=> result\n"
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct_avg:,}\n"
                f"* accuracy: {acc_avg:.2f}%\n"
            )
        else:
            print(
                "=> result\n"
                f"* total: {self._total:,}\n"
                f"* correct_avg: {self._correct_avg:,}\n"
                f"* accuracy_avg: {acc_avg:.2f}%\n"
                f"* correct_min_entropy: {self._correct_min_entropy:,}\n"
                f"* accuracy_min_entropy: {acc_min_entropy:.2f}%\n"
                f"* correct_max_confidence: {self._correct_max_confidence:,}\n"
                f"* accuracy_max_confidence: {acc_max_confidence:.2f}%\n"
            )
            for i in range(len(acc_ind)):
                print(
                    f"* correct_prompt_{i}: {self._correct_ind[i]:,}\n"
                    f"* accuracy_prompt_{i}: {acc_ind[i]:.2f}%\n")

        if self._per_class_res_avg is not None:
            raise NotImplementedError
    
        if self.cfg.TEST.COMPUTE_CMAT:
            raise NotImplementedError
        
        return results


