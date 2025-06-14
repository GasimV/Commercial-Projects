from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
from django.shortcuts import render
from dotenv import load_dotenv
from openai import OpenAI
import os
import re
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ✅ new style


def home(request):
    return render(request, "home.html")


@csrf_exempt
def handle_query(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            user_query = body.get('query', '').strip()

            if not user_query:
                return JsonResponse({"error": "Sorğu boşdur!"}, status=400)

            # GPT Prompt
            schema_context = """
            Sən aşağıdakı cədvəllərlə işləyirsən:
            - customers(id, name, city)
            - orders(id, customer_id, order_date, payment_type ('credit_card', 'cash', 'paypal'), amount)
            - reviews(id, order_id, score)

            Məqsədin: İstifadəçi sorğusuna uyğun SQL sorğusu yaratmaq, cavabı hesablamaq və Azərbaycan dilində nəticəni qısa izah etmək.
            SQL sorğusunu sadə və SQLite3 uyğun saxla.
            """

            # 🔍 Step 1: Generate SQL with GPT-4o
            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": schema_context},
                    {"role": "user", "content": f"İstifadəçi sorğusu: {user_query}"}
                ],
                temperature=0,
            )

            # Raw GPT response
            raw_response = gpt_response.choices[0].message.content.strip()

            # Extract SQL from code block or plain content
            sql_match = re.search(r"```sql\s*(.*?)```", raw_response, re.DOTALL)
            if sql_match:
                sql_code = sql_match.group(1).strip()
            else:
                # Fallback: try to extract the first valid-looking SELECT statement
                for line in raw_response.splitlines():
                    if line.strip().lower().startswith(("select", "with", "insert", "update", "delete")):
                        sql_code = line.strip()
                        break
                else:
                    raise ValueError("LLM cavabında SQL tapılmadı.")

            # 🧠 Step 2: Execute SQL
            with connection.cursor() as cursor:
                cursor.execute(sql_code)
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                result_data = [dict(zip(columns, row)) for row in rows]


            # 📘 Step 3: Generate Summary — include the original query (and SQL) for context
            summary_prompt = (
                f"İstifadəçi sorğusu: {user_query}\n"
                f"İcra edilən SQL: {sql_code}\n"
                f"Nəticələr: {result_data}\n\n"
                "Bu sorğuya uyğun olaraq qısa izah ver:"
            )

            summary_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": schema_context},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )

            summary = summary_response.choices[0].message.content.strip()

            return JsonResponse({
                "sql": sql_code,
                "table": result_data,
                "summary": summary
            })

        except Exception as e:
            import traceback
            print("🔥 Error in handle_query:", traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Only POST requests allowed."}, status=405)
