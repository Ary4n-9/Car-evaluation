import os
import joblib
import pandas as pd
import gradio as gr

# Car Evaluation Prediction System
# Developed By : Aryan
# Roll No      : 241542
# College      : Panipat Institute of Engineering & Technology (PIET), Samalkha

# Load trained model
model = joblib.load("car_safety_model.pkl")

# Prediction Function
def predict_car_safety(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety
):
    data = pd.DataFrame({
        "buying price": [buying_price],
        "maintenance cost": [maintenance_cost],
        "number of doors": [number_of_doors],
        "number of persons": [number_of_persons],
        "lug_boot": [lug_boot],
        "safety": [safety]
    })

    prediction = model.predict(data)[0]

    decision_map = {
        0: "Unacceptable ❌",
        1: "Acceptable ☑️",
        2: "Good 👍",
        3: "Very Good 🌟"
    }

    return decision_map.get(prediction, str(prediction))

# ================================
# Enhanced Gradio UI
# ================================

# Define a custom color palette
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="Car Evaluation Prediction System") as demo:
    
    # 1. Catchy Header Section using HTML
    gr.HTML("""
        <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #4f46e5; font-size: 2.5rem; margin-bottom: 10px;">🚗 Car Evaluation Prediction System</h1>
            <p style="font-size: 1.1rem; color: #6b7280;">
                Predict the overall evaluation and safety of a car based on physical and economic features using Machine Learning.
            </p>
        </div>
    """)

    # 2. Main Layout Structure (Left: Inputs, Right: Output & Info)
    with gr.Row():
        
        # LEFT COLUMN - User Inputs
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Car Specifications")
            
            # Grouping inputs creates a nice visual card around them
            with gr.Group():
                with gr.Row():
                    buying_price = gr.Dropdown(choices=[0, 1, 2, 3], label="💰 Buying Price", info="0 = Low, 3 = Very High")
                    maintenance_cost = gr.Dropdown(choices=[0, 1, 2, 3], label="🔧 Maintenance Cost", info="0 = Low, 3 = Very High")
                
                with gr.Row():
                    number_of_doors = gr.Dropdown(choices=[2, 3, 4, 5], label="🚪 Number of Doors")
                    number_of_persons = gr.Dropdown(choices=[2, 4, 5], label="👥 Passenger Capacity")
                
                with gr.Row():
                    lug_boot = gr.Dropdown(choices=[0, 1, 2], label="🧳 Luggage Boot Size", info="0 = Small, 2 = Big")
                    safety = gr.Dropdown(choices=[0, 1, 2], label="🛡️ Safety Level", info="0 = Low, 2 = High")

            predict_btn = gr.Button("🚀 Predict Car Evaluation", variant="primary", size="lg")

        # RIGHT COLUMN - Results & Developer Info
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Result")
            output = gr.Textbox(
                label="Overall Evaluation",
                show_label=False,
                lines=3,
                text_align="center",
                elem_classes="text-lg font-bold"
            )
            
            # Styled Developer Card
            gr.HTML("""
                <div style="margin-top: 20px; padding: 20px; background-color: #f3f4f6; border-radius: 10px; border-left: 5px solid #4f46e5;">
                    <h3 style="margin-top: 0; color: #111827; font-size: 1.2rem;">👨‍💻 Developer Profile</h3>
                    <p style="margin: 5px 0;"><b>Name:</b> Parth</p>
                    <p style="margin: 5px 0;"><b>Roll No:</b> 241504</p>
                    <p style="margin: 5px 0;"><b>Course:</b> BCA (Data Science)</p>
                    <p style="margin: 5px 0;"><b>College:</b> PIET, Samalkha</p>
                    <hr style="border: 0; height: 1px; background: #e5e7eb; margin: 15px 0;">
                    <p style="margin: 0; font-size: 0.9em; color: #6b7280;"><i>Machine Learning Classification</i></p>
                </div>
            """)

    # 3. Action Click Event
    predict_btn.click(
        fn=predict_car_safety,
        inputs=[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ],
        outputs=output
    )

    # 4. Footer
    gr.HTML("""
        <div style='text-align: center; color: gray; margin-top: 40px; font-size: 0.9em;'>
            © 2026 All Rights Reserved | Developed by Aryan | PIET, Samalkha
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
