from data import *
import pandas as pd
import zoo.attenagg as attenagg
t.manual_seed(15721)

def pipeline_train_validate(model, batch_size, epoch_numb, verbose=False):
    import datetime
    training_data, testing_data = TrainingSet.load_two(0.2)
    dataloader = training_data.to_dataloader(batch_size)
    # training
    model = model.to('cuda:0')
    optim = t.optim.Adam(model.parameters(), lr=1e-5)
    model.train(True)
    for i in range(epoch_numb):
        print("epoch: ", i)
        avg = 0.0
        cnt = 0
        start = datetime.datetime.now()
        for img, txt, label, uid in dataloader:
            loss = -((model(img, txt) + 1e-80) / (1 + 1e-80)).log()
            assert len(loss.shape) == 2
            loss = loss[t.arange(0, batch_size), label].mean()
            loss.backward()
            if verbose: print("loss: ", loss.item())
            optim.step()
            model.zero_grad()
            avg = (avg * cnt + loss.item()) / (cnt + 1)
            cnt += 1
        print("average loss: ", avg)
        end = datetime.datetime.now()
        print("time elapsed: ", end - start)
    # testing
    dataloader = testing_data.to_dataloader(batch_size)
    classification = []
    label_map = ["positive", "neutral", "negative"]
    model.train(False)
    for img, txt, label, uid in dataloader:
        out = model(img, txt).argmax(dim=-1).cpu().detach().numpy()
        uid = uid.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        classification += [{"guid": uid[i], 
                            "out": label_map[out[i]], 
                            "tag": label_map[label[i]]}
                            for i in range(out.shape[0])]
    df = pd.DataFrame(classification)
    df.set_index("guid")
    avg = ((df["out"] == df["tag"]) * 1.0).mean()
    print(avg, flush=True)
    df.to_csv("output.csv", index=False)
    df[df["out"] != df["tag"]].to_csv("test-mistakes.csv", index=False)

def pipeline_train_test(model, batch_size, epoch_numb, verbose=False):
    import datetime
    training_data, testing_data = TrainingSet.load(), TestSet.load()
    dataloader = training_data.to_dataloader(batch_size)
    # training
    model = model.to('cuda:0')
    optim = t.optim.Adam(model.parameters(), lr=1e-5)
    model.train(True)
    for i in range(epoch_numb):
        print("epoch: ", i)
        avg = 0.0
        cnt = 0
        start = datetime.datetime.now()
        for img, txt, label, uid in dataloader:
            loss = -((model(img, txt) + 1e-80) / (1 + 1e-80)).log()
            assert len(loss.shape) == 2
            loss = loss[t.arange(0, batch_size), label].mean()
            loss.backward()
            if verbose: print("loss: ", loss.item())
            optim.step()
            model.zero_grad()
            avg = (avg * cnt + loss.item()) / (cnt + 1)
            cnt += 1
        print("average loss: ", avg)
        end = datetime.datetime.now()
        print("time elapsed: ", end - start)
    dataloader = testing_data.to_dataloader(batch_size)
    # testing
    classification = []
    label_map = ["positive", "neutral", "negative"]
    model.train(False)
    for img, txt, uid in dataloader:
        out = model(img, txt).argmax(dim=-1).cpu().detach().numpy()
        uid = uid.cpu().detach().numpy()
        classification += [{"guid": uid[i], 
                            "tag": label_map[out[i]]}
                            for i in range(out.shape[0])]
    df = pd.DataFrame(classification)
    df.set_index("guid")
    df.to_csv("answer.csv", index=False)

if __name__ == '__main__':
    # import some ... emmm stuff stuff
    import argparse
    import sys
    BATCH_SIZE = 32
    EPOCH_NUMB = 16

    # declare argument parser (AP doesn't always refer to Andrew Pavlo)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-text" , action='store_true')
    ap.add_argument("--only-image", action='store_true')
    ap.add_argument("--get-output", action='store_true')
    ns = ap.parse_args(sys.argv[1:])

    # run
    if ns.get_output:
        pipeline_train_test(attenagg.AttenAGG(), BATCH_SIZE, EPOCH_NUMB)
    elif ns.only_text:
        pipeline_train_validate(attenagg.OnlyImage(), BATCH_SIZE, EPOCH_NUMB)
    elif ns.only_image:
        pipeline_train_validate(attenagg.OnlyText(), BATCH_SIZE, EPOCH_NUMB)
    else:
        pipeline_train_validate(attenagg.AttenAGG(), BATCH_SIZE, EPOCH_NUMB)
