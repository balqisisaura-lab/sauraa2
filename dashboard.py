# ... (kode sebelumnya tetap sama)

# ----------------- GAME ROCK PAPER SCISSORS (IMPLEMENTASI SEDERHANA) -----------------
with tabs[3]:
    st.markdown("<h2 class='section-title'>🎮 Rock Paper Scissors Game</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='text-align: center;'>Main <span style='font-weight: bold; color: #00d4ff;'>Batu-Gunting-Kertas</span> melawan AI! Pilih gesture Anda dan lihat hasilnya.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inisialisasi session state untuk game
    if 'user_choice' not in st.session_state:
        st.session_state['user_choice'] = None
    if 'ai_choice' not in st.session_state:
        st.session_state['ai_choice'] = None
    if 'result' not in st.session_state:
        st.session_state['result'] = None
    if 'score_user' not in st.session_state:
        st.session_state['score_user'] = 0
    if 'score_ai' not in st.session_state:
        st.session_state['score_ai'] = 0
    
    # Fungsi untuk menentukan pemenang
    def determine_winner(user, ai):
        if user == ai:
            return "Draw"
        elif (user == "Rock" and ai == "Scissors") or \
             (user == "Paper" and ai == "Rock") or \
             (user == "Scissors" and ai == "Paper"):
            return "You Win"
        else:
            return "AI Wins"
    
    # Fungsi untuk reset game
    def reset_game():
        st.session_state['user_choice'] = None
        st.session_state['ai_choice'] = None
        st.session_state['result'] = None
    
    # Layout game
    col_game_left, col_game_center, col_game_right = st.columns([1, 2, 1])
    
    with col_game_center:
        st.markdown("### Pilih Gesture Anda:")
        
        # Tombol pilihan
        col_rock, col_paper, col_scissors = st.columns(3)
        
        with col_rock:
            if st.button("✊ Rock", key="rock_btn"):
                st.session_state['user_choice'] = "Rock"
                st.session_state['ai_choice'] = np.random.choice(["Rock", "Paper", "Scissors"])
                st.session_state['result'] = determine_winner(st.session_state['user_choice'], st.session_state['ai_choice'])
                if st.session_state['result'] == "You Win":
                    st.session_state['score_user'] += 1
                elif st.session_state['result'] == "AI Wins":
                    st.session_state['score_ai'] += 1
        
        with col_paper:
            if st.button("✋ Paper", key="paper_btn"):
                st.session_state['user_choice'] = "Paper"
                st.session_state['ai_choice'] = np.random.choice(["Rock", "Paper", "Scissors"])
                st.session_state['result'] = determine_winner(st.session_state['user_choice'], st.session_state['ai_choice'])
                if st.session_state['result'] == "You Win":
                    st.session_state['score_user'] += 1
                elif st.session_state['result'] == "AI Wins":
                    st.session_state['score_ai'] += 1
        
        with col_scissors:
            if st.button("✌ Scissors", key="scissors_btn"):
                st.session_state['user_choice'] = "Scissors"
                st.session_state['ai_choice'] = np.random.choice(["Rock", "Paper", "Scissors"])
                st.session_state['result'] = determine_winner(st.session_state['user_choice'], st.session_state['ai_choice'])
                if st.session_state['result'] == "You Win":
                    st.session_state['score_user'] += 1
                elif st.session_state['result'] == "AI Wins":
                    st.session_state['score_ai'] += 1
        
        # Tombol reset
        if st.button("🔄 Reset Game", key="reset_btn"):
            reset_game()
        
        # Tampilkan hasil
        if st.session_state['user_choice'] and st.session_state['ai_choice']:
            st.markdown("---")
            st.markdown("### Hasil:")
            
            col_user, col_vs, col_ai = st.columns([1, 1, 1])
            
            with col_user:
                st.markdown(f"**Anda:** {st.session_state['user_choice']}")
                if st.session_state['user_choice'] == "Rock":
                    st.markdown("✊")
                elif st.session_state['user_choice'] == "Paper":
                    st.markdown("✋")
                else:
                    st.markdown("✌")
            
            with col_vs:
                st.markdown("**VS**")
            
            with col_ai:
                st.markdown(f"**AI:** {st.session_state['ai_choice']}")
                if st.session_state['ai_choice'] == "Rock":
                    st.markdown("✊")
                elif st.session_state['ai_choice'] == "Paper":
                    st.markdown("✋")
                else:
                    st.markdown("✌")
            
            # Hasil akhir
            if st.session_state['result'] == "You Win":
                st.success(f"🎉 {st.session_state['result']}!")
            elif st.session_state['result'] == "AI Wins":
                st.error(f"😢 {st.session_state['result']}!")
            else:
                st.info(f"🤝 {st.session_state['result']}!")
        
        # Skor
        st.markdown("---")
        st.markdown("### Skor:")
        col_score_user, col_score_ai = st.columns(2)
        
        with col_score_user:
            st.metric("Anda", st.session_state['score_user'])
        
        with col_score_ai:
            st.metric("AI", st.session_state['score_ai'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Tips:** Klik salah satu gesture untuk bermain! AI akan memilih secara acak.")

# ... (kode setelahnya tetap sama)
