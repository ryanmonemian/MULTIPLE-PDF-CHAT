css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.chat-message .avatar div {
  width: 78px;
  height: 78px;
  border-radius: 50%;
  font-size: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #ffffff20; /* Slight white tint circle background */
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <div>📚</div>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <div>👤</div>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''