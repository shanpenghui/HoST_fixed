from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

ea = event_accumulator.EventAccumulator("./events.out.tfevents.1753631998.unitree.29062.0")
ea.Reload()

for tag in ea.Tags()["scalars"]:
    events = ea.Scalars(tag)
    df = pd.DataFrame([(e.step, e.value) for e in events], columns=["step", tag])
    df.to_csv(f"{tag.replace('/', '_')}.csv", index=False)

