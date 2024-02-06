from base import FraudScreener, FraudCriteria
import streamlit as st
import tempfile
0
st.title('Fraud Screener')
st.write('Enter your report here')

uploaded_report = st.file_uploader("Upload your report", type=["pdf", "txt"])

if uploaded_report is not None:
    mime_type = uploaded_report.type
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_report.read())
        tmp_file_path = tmp.name
        with open(tmp_file_path, 'rb') as f:
                print(f.read())
            
    criteria = FraudCriteria()

    fraud_screener = FraudScreener(
        uploaded_report=tmp_file_path,
        fraud_criteria=criteria.FRAUD_CRITERIA,
        mime_type=mime_type
    )
    with st.spinner("Processing report..."):
        response = fraud_screener.run(uploaded_report=tmp_file_path)
    
   
    if response is not None:
        yes_count = sum(decision.decision for decision in response.criteria_decisions)
        if yes_count > 2:
            st.markdown("<h1 style='text-align: center; color: red;'>RECOMMEND FOLLOW-UP</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: red;'>PASS</h1>", unsafe_allow_html=True)

        st.markdown(f"**Overall Reasoning:**\n{response.overall_reasoning}\n")
        st.divider()
        decisions_md = "\n**Criteria Decisions:**\n"
        for decision in response.criteria_decisions:
            decisions_md += f"- Criterion: {decision.criterion}\n"
            decisions_md += f"  - Potential Fraud: {'Yes' if decision.decision else 'No'}\n"
            decisions_md += f"  - Reasoning: {decision.reasoning}\n\n\n\n"  
        st.markdown(decisions_md)
        st.divider()
    else:
        st.markdown("No response")

    