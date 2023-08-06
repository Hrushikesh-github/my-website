from celery import shared_task


VISITS_DB_FILE = "visits_db.txt"


@shared_task
def get_counts():
    import time
    with open(VISITS_DB_FILE, "r") as file:
        lines = file.readlines()

    time.sleep(5)
    # Count the number of existing visits
    visit_count = len(lines)
    return visit_count


@shared_task
def increment_visit_count():
    from datetime import datetime
    import time
    # Read the existing visit records from the file
    time.sleep(5)

    with open(VISITS_DB_FILE, "r") as file:
        lines = file.readlines()

    # Count the number of existing visits
    visit_count = len(lines)

    # Increment the visit count by 1
    visit_count += 1

    # Append the new visit record to the file
    with open(VISITS_DB_FILE, "a") as file:
        file.write(f"{visit_count},{datetime.now()},\n")

    return visit_count

@shared_task
def simplee_task():
    import time
    time.sleep(10)
    print("CEELRY TASK IS COMPLETED")
    return {"visits": 4}
