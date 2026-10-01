import csv
import itertools
import math
import time
from collections import Counter
import matplotlib.pyplot as plt


def read_data(filename):
    with open(filename, encoding="cp1251", newline="") as f:
        return [
            {x.strip() for x in row if x.strip()}
            for row in csv.reader(f)
            if any(x.strip() for x in row)
        ]


def apriori(transactions, min_support, sort_by="support"):
    n = len(transactions)
    min_count = math.ceil(n * min_support)

    counts = Counter(item for basket in transactions for item in basket)

    current = {
        frozenset([item]): count
        for item, count in counts.items()
        if count >= min_count
    }

    result = dict(current)
    k = 2

    while current:
        previous = set(current)
        candidates = set()

        for a, b in itertools.combinations(previous, 2):
            candidate = a | b
            if len(candidate) == k and all(
                frozenset(s) in previous
                for s in itertools.combinations(candidate, k - 1)
            ):
                candidates.add(candidate)

        if not candidates:
            break

        counts = Counter(
            candidate
            for basket in transactions
            for candidate in candidates
            if candidate.issubset(basket)
        )

        current = {
            candidate: count
            for candidate, count in counts.items()
            if count >= min_count
        }

        result.update(current)
        k += 1

    if sort_by == "support":
        result = sorted(
            result.items(),
            key=lambda x: (-x[1] / n, tuple(sorted(x[0])))
        )
    else:
        result = sorted(
            result.items(),
            key=lambda x: tuple(sorted(x[0]))
        )

    return [
        (tuple(sorted(items)), count / n)
        for items, count in result
    ]


def generate_rules(frequent_itemsets, min_confidence, sort_by="confidence"):
    rules = []
    support_dict = {
        frozenset(items): support
        for items, support in frequent_itemsets
    }

    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue

        itemset_frozen = frozenset(itemset)

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent_frozen = frozenset(antecedent)
                consequent = itemset_frozen - antecedent_frozen

                antecedent_support = support_dict.get(antecedent_frozen, 0)

                if antecedent_support > 0:
                    confidence = support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append({
                            'antecedent': tuple(sorted(antecedent)),
                            'consequent': tuple(sorted(consequent)),
                            'support': support,
                            'confidence': confidence
                        })

    if sort_by == "confidence":
        rules.sort(key=lambda x: (-x['confidence'], -x['support']))
    elif sort_by == "support":
        rules.sort(key=lambda x: (-x['support'], -x['confidence']))
    else:
        rules.sort(key=lambda x: (x['antecedent'], x['consequent']))

    return rules


def run_experiments(transactions, min_support=0.01):
    # Адаптированный диапазон уверенности под реалии продуктового набора данных
    confidences = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    results = []

    frequent_itemsets = apriori(transactions, min_support)

    for conf in confidences:
        start = time.perf_counter()
        rules = generate_rules(frequent_itemsets, conf)
        elapsed = time.perf_counter() - start

        results.append({
            "confidence": conf,
            "time": elapsed,
            "total_rules": len(rules)
        })

    return results


def save_rules_to_csv(rules, filename="rules.csv"):
    with open(filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Антецедент", "Консеквент", "Поддержка", "Уверенность"])
        for rule in rules:
            writer.writerow([
                ", ".join(rule['antecedent']),
                ", ".join(rule['consequent']),
                f"{rule['support']:.4f}",
                f"{rule['confidence']:.4f}"
            ])


def make_charts(results):
    x = [f"{r['confidence'] * 100:.0f}%" for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(x, [r["time"] for r in results], color='skyblue', edgecolor='black')
    plt.xlabel("Порог уверенности")
    plt.ylabel("Время выполнения, с")
    plt.title("Время выполнения поиска ассоциативных правил")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("rules_time_vs_confidence.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(x, [r["total_rules"] for r in results], color='lightgreen', edgecolor='black')
    plt.xlabel("Порог уверенности")
    plt.ylabel("Количество правил")
    plt.title("Количество найденных ассоциативных правил")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("rules_count_vs_confidence.png", dpi=300)
    plt.show()


def print_short_rules(rules, max_total_length=7):
    short_rules = [
        rule for rule in rules
        if len(rule['antecedent']) + len(rule['consequent']) <= max_total_length
    ]

    print(f"\nПравила с суммарной длиной <= {max_total_length} ({len(short_rules)} шт.):")
    for i, rule in enumerate(short_rules[:20], 1):
        ant = ", ".join(rule['antecedent'])
        con = ", ".join(rule['consequent'])
        print(f"{i:3}. {{{ant}}} -> {{{con}}} | Supp: {rule['support']:.4f}, Conf: {rule['confidence']:.4f}")


def main():
    transactions = read_data("baskets.csv")
    
    min_support = 0.01  # Снижено до 1% для поиска узких, но сильных ассоциаций
    results = run_experiments(transactions, min_support)

    with open("rules_experiment_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Порог уверенности", "Время, с", "Количество правил"])
        for r in results:
            writer.writerow([f"{r['confidence']*100:.0f}%", f"{r['time']:.6f}", r['total_rules']])

    make_charts(results)

    frequent_itemsets = apriori(transactions, min_support)
    rules = generate_rules(frequent_itemsets, 0.30) # Берем минимальный порог для демонстрации

    save_rules_to_csv(rules, "all_rules_30.csv")
    print_short_rules(rules)


if __name__ == "__main__":
    main()