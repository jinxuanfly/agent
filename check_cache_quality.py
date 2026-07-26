import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

checkpoint_dir = 'checkpoints/hateful_memes/'

print("=" * 70)
print("Cache Quality Check")
print("=" * 70)

for split in ['train', 'val']:
    print("\n[%s set]" % split)
    for agent in range(3):
        path = os.path.join(checkpoint_dir, 'llm_%s_agent%d.pt' % (split, agent))
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
            beliefs = data['beliefs']
            uncertainties = data['uncertainties']
            
            pred0 = (beliefs.argmax(dim=-1) == 0).sum().item()
            pred1 = (beliefs.argmax(dim=-1) == 1).sum().item()
            total = beliefs.shape[0]
            
            fallback_count = 0
            for i in range(total):
                pred = beliefs[i].argmax().item()
                conf = beliefs[i][pred].item()
                if pred == 0 and abs(conf - 0.5) < 0.01:
                    fallback_count += 1
            
            avg_u = uncertainties.mean().item()
            avg_conf = beliefs.max(dim=-1).values.mean().item()
            
            pct_fallback = fallback_count * 100.0 / total
            
            print("  Agent%d:" % (agent+1))
            print("    Samples: %d" % total)
            print("    Predictions: Label0=%d (%f), Label1=%d (%f)" % 
                  (pred0, pred0*100.0/total, pred1, pred1*100.0/total))
            print("    Fallback samples: %d (%f)" % (fallback_count, pct_fallback))
            print("    Avg uncertainty: %f" % avg_u)
            print("    Avg confidence: %f" % avg_conf)
            
            if fallback_count > total * 0.5:
                print("    [WARN] Fallback high, Agent%d may be invalid" % (agent+1))
            elif avg_conf < 0.6:
                print("    [WARN] Low confidence, results may be unreliable")
        else:
            print("  Agent%d: Cache file not found" % (agent+1))