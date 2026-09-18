```python
import os
import joblib
import pandas as pd
import gradio as gr

# ============================================================
# CAR EVALUATION PREDICTION SYSTEM
# Developed By: Aryan
# Roll No: 241542
# ============================================================

# Load trained model
model = joblib.load("car_safety_model.pkl")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

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
        0: "❌ Unacceptable",
        1: "⚠️ Acceptable",
        2: "👍 Good",
        3: "🏆 Very Good"
    }

    return decision_map.get(prediction, str(prediction))


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_form():
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        ""
    )


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

/* Main background */
.gradio-container {
    background:
        linear-gradient(rgba(8, 15, 30, 0.88), rgba(8, 15, 30, 0.94)),
        url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=2000&q=90")
        center center / cover fixed !important;
    font-family: Arial, sans-serif;
}


/* Main wrapper */
.main-container {
    max-width: 1150px !important;
    margin: auto !important;
}


/* Hero section */
.hero {
    text-align: center;
    padding: 45px 20px 35px 20px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-bottom: 12px;
    letter-spacing: -1px;
}

.hero p {
    color: #cbd5e1;
    font-size: 18px;
    max-width: 750px;
    margin: auto;
    line-height: 1.7;
}


/* Glass cards */
.glass-card {
    background: rgba(255, 255, 255, 0.09) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 24px !important;
    padding: 28px !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
}


/* Section headings */
.section-title {
    color: white;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #cbd5e1;
    margin-bottom: 20px;
}


/* Labels */
label span {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}


/* Dropdown */
select,
input {
    border-radius: 12px !important;
}


/* Predict button */
.predict-btn {
    height: 58px !important;
    border-radius: 15px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    margin-top: 15px;
    transition: all 0.25s ease;
}

.predict-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.35);
}


/* Reset button */
.reset-btn {
    height: 52px !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
}


/* Output */
.result-box textarea {
    font-size: 25px !important;
    font-weight: 800 !important;
    text-align: center !important;
    min-height: 80px !important;
}


/* Info cards */
.info-card {
    text-align: center;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 20px;
    color: white;
}

.info-card .icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.info-card h3 {
    margin: 5px;
    font-size: 17px;
}

.info-card p {
    color: #cbd5e1;
    font-size: 13px;
}


/* Footer */
.footer {
    text-align: center;
    padding: 30px 10px;
    color: #94a3b8;
    font-size: 14px;
}

.footer strong {
    color: #e2e8f0;
}


/* Mobile */
@media (max-width: 700px) {

    .hero h1 {
        font-size: 34px;
    }

    .hero p {
        font-size: 15px;
    }

    .glass-card {
        padding: 18px !important;
    }
}

"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="Smart Car Evaluation System"
) as demo:

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="hero">

            <div style="
                font-size:60px;
                margin-bottom:10px;
            ">
                🚗
            </div>

            <h1>Smart Car Evaluation</h1>

            <p>
                An intelligent machine learning system that evaluates
                the overall quality and safety of a car based on its
                price, maintenance, capacity, luggage space and safety.
            </p>

        </div>
        """
    )


    # --------------------------------------------------------
    # INFORMATION CARDS
    # --------------------------------------------------------

    with gr.Row():

        gr.HTML(
            """
            <div class="info-card">
                <div class="icon">🤖</div>
                <h3>Machine Learning</h3>
                <p>ML-based classification model</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <div class="icon">⚡</div>
                <h3>Instant Prediction</h3>
                <p>Get results within seconds</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <div class="icon">🛡️</div>
                <h3>Safety Analysis</h3>
                <p>Evaluate important car factors</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <div class="icon">📊</div>
                <h3>Smart Evaluation</h3>
                <p>Four-level evaluation system</p>
            </div>
            """
        )


    gr.Markdown("<br>")


    # --------------------------------------------------------
    # PREDICTION CARD
    # --------------------------------------------------------

    with gr.Column(elem_classes="glass-card"):

        gr.HTML(
            """
            <div class="section-title">
                🔍 Enter Car Details
            </div>

            <div class="section-subtitle">
                Select the specifications of the car to generate
                an intelligent evaluation.
            </div>
            """
        )


        # Row 1
        with gr.Row():

            buying_price = gr.Dropdown(
                choices=[
                    ( "Low", 0),
                    ( "Medium", 1),
                    ( "High", 2),
                    ( "Very High", 3)
                ],
                label="💰 Buying Price",
                info="Select the car's buying price level"
            )

            maintenance_cost = gr.Dropdown(
                choices=[
                    ("Low", 0),
                    ("Medium", 1),
                    ("High", 2),
                    ("Very High", 3)
                ],
                label="🔧 Maintenance Cost",
                info="Select expected maintenance cost"
            )


        # Row 2
        with gr.Row():

            number_of_doors = gr.Dropdown(
                choices=[
                    ("2 Doors", 2),
                    ("3 Doors", 3),
                    ("4 Doors", 4),
                    ("5 Doors", 5)
                ],
                label="🚪 Number of Doors",
                info="Number of doors in the car"
            )

            number_of_persons = gr.Dropdown(
                choices=[
                    ("2 Persons", 2),
                    ("4 Persons", 4),
                    ("5 Persons", 5)
                ],
                label="👥 Passenger Capacity",
                info="Maximum passenger capacity"
            )


        # Row 3
        with gr.Row():

            lug_boot = gr.Dropdown(
                choices=[
                    ("Small", 0),
                    ("Medium", 1),
                    ("Large", 2)
                ],
                label="🧳 Luggage Boot",
                info="Select luggage boot capacity"
            )

            safety = gr.Dropdown(
                choices=[
                    ("Low", 0),
                    ("Medium", 1),
                    ("High", 2)
                ],
                label="🛡️ Safety Level",
                info="Select the car's safety level"
            )


        # Buttons
        with gr.Row():

            predict_btn = gr.Button(
                "🚀 Evaluate My Car",
                variant="primary",
                elem_classes="predict-btn"
            )

            reset_btn = gr.Button(
                "🔄 Reset",
                variant="secondary",
                elem_classes="reset-btn"
            )


        # Result
        output = gr.Textbox(
            label="🏆 Evaluation Result",
            placeholder="Your car evaluation will appear here...",
            elem_classes="result-box",
            interactive=False
        )


    # --------------------------------------------------------
    # EVALUATION GUIDE
    # --------------------------------------------------------

    gr.Markdown("<br>")

    with gr.Column(elem_classes="glass-card"):

        gr.HTML(
            """
            <div class="section-title">
                📋 Evaluation Guide
            </div>

            <div style="
                color:#cbd5e1;
                line-height:1.8;
                font-size:15px;
            ">

            <p>
            <b style="color:white;">❌ Unacceptable</b>
            — The car does not satisfy the required evaluation criteria.
            </p>

            <p>
            <b style="color:white;">⚠️ Acceptable</b>
            — The car meets basic evaluation requirements.
            </p>

            <p>
            <b style="color:white;">👍 Good</b>
            — The car has generally good characteristics.
            </p>

            <p>
            <b style="color:white;">🏆 Very Good</b>
            — The car performs very well across the evaluated factors.
            </p>

            </div>
            """
        )


    # --------------------------------------------------------
    # BUTTON EVENTS
    # --------------------------------------------------------

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


    reset_btn.click(
        fn=reset_form,
        inputs=[],
        outputs=[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety,
            output
        ]
    )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="footer">

            <hr style="
                border:0;
                border-top:1px solid rgba(255,255,255,0.12);
                margin-bottom:25px;
            ">

            <div style="font-size:25px; margin-bottom:10px;">
                🚗
            </div>

            <strong>Smart Car Evaluation Prediction System</strong>

            <p>
                Machine Learning Classification Project
            </p>

            <p>
                Developed by <strong>Parth</strong> |
                Roll No. <strong>241504</strong>
            </p>

            <p>
                BCA (Data Science) |
                Panipat Institute of Engineering & Technology (PIET), Samalkha
            </p>

            <p style="font-size:12px;">
                © 2026 Smart Car Evaluation System • All Rights Reserved
            </p>

        </div>
        """
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
```
