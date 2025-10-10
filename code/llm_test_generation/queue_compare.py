#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import sys

QUEUE_LABELS = [
    "CTI",
    "DFIR::incidents",
    "DFIR::phishing",
    "OFFSEC::Pentesting",
    "OFFSEC::CVD",
    "SMS",
    "Trash"
]

def gpt_queue(title, content, client):
    prompt = (
        "You are a CERT ticket classifier. "
        "Given the ticket's title and content, assign it to one of these queues: "
        f"{', '.join(QUEUE_LABELS)}.\n"
        "Respond ONLY with the most appropriate queue label.\n"
        f"Title: {title}\nContent: {content}\nQueue:"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies CERT tickets."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )
    label = response.choices[0].message.content.strip().split()[0]
    return label

def run_llm_prediction(args):
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.azure_key,
        api_version=args.azure_version,
    )
    infile = args.infile
    outfile = args.llm_preds
    print(f"Running LLM predictions from '{infile}' to '{outfile}' ...")
    with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            ticket = json.loads(line)
            label = gpt_queue(ticket.get("title", ""), ticket.get("content", ""), client)
            ticket["llm_queue"] = label
            fout.write(json.dumps(ticket, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"  Processed {i} tickets...", end="\r")
    print(f"\nDone! LLM predictions written to {outfile}")

def load_jsonl(filename):
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            ticket_id = record.get('ticket_id')
            if ticket_id:
                data[ticket_id] = record
    return data

def compare_queues(llm_pred_file, model_pred_file):
    llm_predictions = load_jsonl(llm_pred_file)
    model_predictions = load_jsonl(model_pred_file)
    common_ids = set(llm_predictions.keys()) & set(model_predictions.keys())
    print(f"LLM predictions file: {len(llm_predictions)} records")
    print(f"Model predictions file: {len(model_predictions)} records")
    print(f"Common ticket_ids: {len(common_ids)}")
    if not common_ids:
        print("No common ticket_ids found!")
        return
    matches, mismatches, comparison_results = 0, 0, []
    for ticket_id in common_ids:
        llm_queue = llm_predictions[ticket_id].get('llm_queue', 'N/A')
        assigned_queue = model_predictions[ticket_id].get('assigned_queue', 'N/A')
        is_match = llm_queue == assigned_queue
        matches += int(is_match)
        mismatches += int(not is_match)
        comparison_results.append({
            'ticket_id': ticket_id,
            'ground_truth': llm_queue,
            'prediction': assigned_queue,
            'match': is_match
        })
    total = len(common_ids)
    accuracy = matches / total if total > 0 else 0
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Total comparisons: {total}")
    print(f"Matches: {matches} ({accuracy*100:.2f}%)")
    print(f"Mismatches: {mismatches} ({(1-accuracy)*100:.2f}%)")
    mismatched_results = [r for r in comparison_results if not r['match']]
    if mismatched_results:
        print(f"\nFirst 20 mismatches:")
        for result in mismatched_results[:20]:
            print(f"Ticket {result['ticket_id']}: LLM='{result['ground_truth']}' vs Model='{result['prediction']}'")
    ground_truth_counts, prediction_counts = defaultdict(int), defaultdict(int)
    for result in comparison_results:
        ground_truth_counts[result['ground_truth']] += 1
        prediction_counts[result['prediction']] += 1
    print("\nLLM Distribution:")
    for queue, count in sorted(ground_truth_counts.items()):
        print(f"  {queue}: {count}")
    print("Model Distribution:")
    for queue, count in sorted(prediction_counts.items()):
        print(f"  {queue}: {count}")
    df = pd.DataFrame(comparison_results)
    df.to_csv('queue_comparison_results.csv', index=False)
    print("Saved detailed results to 'queue_comparison_results.csv'")
    if mismatched_results:
        mismatch_data = []
        for result in mismatched_results:
            ticket_id = result['ticket_id']
            llm_record = llm_predictions.get(ticket_id, {})
            test_record = model_predictions.get(ticket_id, {})
            mismatch_entry = dict(llm_record)
            mismatch_entry.update({
                'ground_truth_llm': result['ground_truth'],
                'prediction_assigned': result['prediction'],
                'mismatch': True
            })
            for key, value in test_record.items():
                if key != 'ticket_id':
                    mismatch_entry[f'assigned_{key}'] = value
            mismatch_data.append(mismatch_entry)
        mismatch_df = pd.DataFrame(mismatch_data)
        mismatch_df.to_csv('mismatch.csv', index=False)
        print(f"Mismatches saved to 'mismatch.csv' ({len(mismatched_results)} records)")
        with open('mismatch.txt', 'w', encoding='utf-8') as f:
            f.write(f"MISMATCH ANALYSIS - {len(mismatched_results)} cases\n")
            f.write("=" * 80 + "\n\n")
            for i, result in enumerate(mismatched_results, 1):
                ticket_id = result['ticket_id']
                llm_record = llm_predictions.get(ticket_id, {})
                f.write(f"[{ticket_id}], [{result['prediction']}], [{result['ground_truth']}]\n\n")
                f.write(f"{llm_record.get('content', 'No content available')}\n")
                if i < len(mismatched_results):
                    f.write("\n" + "-" * 80 + "\n\n")
        print(f"Readable format saved to 'mismatch.txt' ({len(mismatched_results)} cases)")
    print("\n=== PER-CATEGORY METRICS ===")
    category_metrics = {}
    all_categories = set(r['ground_truth'] for r in comparison_results)
    for category in sorted(all_categories):
        cat_results = [r for r in comparison_results if r['ground_truth'] == category]
        total_in_cat = len(cat_results)
        correct_pred = sum(1 for r in cat_results if r['match'])
        assigned_pred_this_cat = [r for r in comparison_results if r['prediction'] == category]
        assigned_correct = sum(1 for r in assigned_pred_this_cat if r['match'])
        precision = assigned_correct / len(assigned_pred_this_cat) if assigned_pred_this_cat else 0
        recall = correct_pred / total_in_cat if total_in_cat > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        category_metrics[category] = {
            'total_instances': total_in_cat,
            'correct_predictions': correct_pred,
            'incorrect_predictions': total_in_cat - correct_pred,
            'accuracy': correct_pred / total_in_cat if total_in_cat > 0 else 0,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        print(f"{category}: Acc={category_metrics[category]['accuracy']:.3f} Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f}")
    pd.DataFrame.from_dict(category_metrics, orient='index').to_csv('category_metrics.csv')
    print("Category metrics saved to 'category_metrics.csv'")
    y_true = [r['ground_truth'] for r in comparison_results]
    y_pred = [r['prediction'] for r in comparison_results]
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\n=== CONFUSION MATRIX ===")
    categories_list = sorted(list(all_categories))
    cm = confusion_matrix(y_true, y_pred, labels=categories_list)
    print("Rows = Actual, Columns = Predicted")
    print(" " * 14 + "".join(f"{cat[:10]:>10}" for cat in categories_list))
    for i, actual_cat in enumerate(categories_list):
        print(f"{actual_cat[:14]:14}" + "".join(f"{cm[i][j]:10d}" for j in range(len(categories_list))))
    print("\nDONE.")

def main():
    parser = argparse.ArgumentParser(description="LLM vs Model Queue Classification Comparator")
    parser.add_argument("--llm_preds", type=str, required=True, help="Path to JSONL with LLM predictions")
    parser.add_argument("--model_preds", type=str, required=True, help="Path to JSONL with model/classifier predictions")
    parser.add_argument("--run_llm", action='store_true', help="Run LLM prediction first (requires Azure API info)")
    parser.add_argument("--infile", type=str, help="Source file for LLM predictions (if --run_llm)")
    parser.add_argument("--azure_endpoint", type=str, default="", help="Azure OpenAI endpoint")
    parser.add_argument("--azure_key", type=str, default="", help="Azure OpenAI API key")
    parser.add_argument("--azure_version", type=str, default="2024-02-15-preview", help="Azure OpenAI API version")
    args = parser.parse_args()
    if args.run_llm:
        if not (args.infile and args.llm_preds and args.azure_endpoint and args.azure_key):
            print("If running LLM, provide --infile, --llm_preds, --azure_endpoint, and --azure_key")
            sys.exit(1)
        run_llm_prediction(args)
        print("\nLLM predictions done. Proceeding to comparison...\n")
    compare_queues(args.llm_preds, args.model_preds)

if __name__ == "__main__":
    main()
