
def load_file(filepath):

    annotations = {}

    for line in open(filepath):
        record, string, entity = line.strip().split('\t', 2)
        annotations[(record, string)] = entity

    return annotations

def get_performance (gold_file,pred_file):

    # Load gold annotations and predictions
    gold = load_file(gold_file)
    pred = load_file(pred_file)

    n_gold = len(gold)
    n_predicted = len(pred)
    print('gold: %s' % n_gold)
    print('predicted: %s' % n_predicted)

    # Evaluate predictions
    n_correct = sum( int(pred[i]==gold[i]) for i in set(gold) & set(pred) )
    print('correct: %s' % n_correct)

    # Calculate scores
    precision = float(n_correct) / float(n_predicted)
    print('precision: %s' % precision )
    recall = float(n_correct) / float(n_gold)
    print('recall: %s' % recall )
    f1 = 2 * ( (precision * recall) / (precision + recall) )
    print('f1: %s' % f1 )