# sample for tool calling

def get_customer_info(customer_id):
    query = "SELECT name, email, phone FROM customers WHERE id = %s"
    result = db.execute(query, (customer_id,)).fetchone()
    return dict(result) if result else "Customer not found"