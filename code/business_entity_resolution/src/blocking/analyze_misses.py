import pandas as pd


GROUND_TRUTH = r"C:\Users\Y Chandrakala\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv"

CANDIDATES = r"output/candidate_pairs_sample.tsv"

S1_PATH = r"C:\Users\Y Chandrakala\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv"

S2_PATH = r"C:\Users\Y Chandrakala\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv"

S3_PATH = r"C:\Users\Y Chandrakala\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv"


gt = pd.read_csv(GROUND_TRUTH, sep="\t").fillna("")

cand = pd.read_csv(CANDIDATES, sep="\t").fillna("")


s1 = pd.read_csv(S1_PATH, sep="\t").set_index("entity_id")
s2 = pd.read_csv(S2_PATH, sep="\t").set_index("entity_id")
s3 = pd.read_csv(S3_PATH, sep="\t").set_index("entity_id")


cand_dict = dict(
    zip(
        cand["source1_entity_id"],
        cand["candidate_entity_ids"]
    )
)


count = 0

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in cand_dict:
        continue

    generated = set(
        cand_dict[s1_id].split(",")
    ) if cand_dict[s1_id] else set()


    true_matches = set(
        row["matched_entity_ids"].split(",")
    ) if row["matched_entity_ids"] else set()


    missed = true_matches - generated


    if missed:

        print("\n======================")
        print("S1:", s1_id)

        print("\nName:")
        print(s1.loc[s1_id]["business_name"])

        print("\nAddress:")
        print(s1.loc[s1_id]["business_address"])

        print("\nMISSED MATCHES:")

        for mid in list(missed)[:3]:

            if mid.startswith("S2"):
                x = s2.loc[mid]
            else:
                x = s3.loc[mid]

            print("\nID:", mid)
            print("Name:", x["business_name"])
            print("Address:", x["business_address"])

        count += 1

        if count == 10:
            break


print("\nTotal missed examples shown:", count)