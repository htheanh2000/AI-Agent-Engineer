"""
function_calling_basic.py

Demo Function Calling đơn giản cho Bài 2:
- 2 tool: get_weather, get_stock_price
- Agent tự chọn tool
- Không dùng top_k hoặc kỹ thuật nâng cao
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================================
# 1. Định nghĩa 2 function cho Agent
# ======================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Lấy thời tiết theo thành phố",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Tên thành phố"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Lấy giá cổ phiếu theo mã",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Mã cổ phiếu cần tra cứu"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]


# ======================================
# 2. Code giả lập API thật
# (Trong bài học chưa cần gọi API ngoài)
# ======================================

def get_weather(city: str):
    # Demo dữ liệu mock
    mock = {
        "Hanoi": "Trời nhiều mây, 23°C",
        "HCM": "Nắng nhẹ, 30°C",
        "Tokyo": "Có mưa, 18°C"
    }
    return mock.get(city, "Không tìm thấy dữ liệu thời tiết.")


def get_stock_price(symbol: str):
    mock = {
        "AAPL": "187.23 USD",
        "TSLA": "242.51 USD",
        "BTC": "92,000 USD"
    }
    return mock.get(symbol, "Không tìm thấy dữ liệu cổ phiếu.")


# ======================================
# 3. Xử lý function_call của Agent
# ======================================

def handle_tool_call(tool_call):
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "get_weather":
        city = arguments["city"]
        result = get_weather(city)
        return result

    if function_name == "get_stock_price":
        symbol = arguments["symbol"]
        result = get_stock_price(symbol)
        return result

    return "Function không tồn tại."


# ======================================
# 4. Vòng lặp Function Calling
# ======================================

def ask_agent(user_input: str, model="gpt-4.1-mini"):
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là Agent real-time. "
                "Nhiệm vụ của bạn là dùng FUNCTION CALLING để lấy thông tin. "
                "Bạn chỉ trả lời dựa trên dữ liệu từ function."
            )
        },
        {"role": "user", "content": user_input}
    ]

    # ----------- Lần gọi 1: Model quyết định có gọi function hay không -----------
    first_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    msg = first_response.choices[0].message

    # Nếu không gọi hàm → trả lời luôn
    if not msg.tool_calls:
        return msg.content

    # Nếu có tool_call
    tool_call = msg.tool_calls[0]
    tool_result = handle_tool_call(tool_call)

    messages.append(msg)  # thêm tool_call vào lịch sử

    # Trả kết quả tool lại cho model
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": tool_result
    })

    # ----------- Lần gọi 2: Model tạo câu trả lời cuối dựa trên tool_result -----------
    final_response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return final_response.choices[0].message.content


# ======================================
# 5. Demo
# ======================================

if __name__ == "__main__":
    print("=== Function Calling Demo (Bài 2) ===")
    while True:
        q = input("\nBạn hỏi gì? (Enter để thoát): ").strip()
        if not q:
            break
        answer = ask_agent(q)
        print("\n[Agent]:", answer)
