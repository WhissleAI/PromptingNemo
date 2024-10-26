document.getElementById("send-btn").addEventListener("click", function() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    // Display the user's message in the chat box
    displayMessage(userInput, "user-message");

    // Send the user's input to the FastAPI server
    fetch("/query/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Display the bot's response in the chat box, excluding the YouTube links
        const cleanedResponse = data.chatgpt_response.replace(/https?:\/\/[^\s]+/g, '');
        displayMessage(cleanedResponse, "bot-message");

        // Clear the video container before adding new videos
        clearVideoContainer();

        // Embed all YouTube videos below the chat box, if available
        if (data.youtube_links && data.youtube_links.length > 0) {
            data.youtube_links.forEach(link => {
                embedYouTubeVideos(link);
            });
        } else {
            console.error("No YouTube links found in the response.");
        }
    })
    .catch(error => console.error("Error:", error));

    // Clear the input field
    document.getElementById("user-input").value = "";
});

function displayMessage(message, className) {
    const chatBox = document.getElementById("chat-box");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${className}`;
    
    const icon = document.createElement("i");
    icon.className = `icon ${className === "user-message" ? "fas fa-user" : "fas fa-robot"}`;
    
    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.innerHTML = message;
    
    messageDiv.appendChild(icon);
    messageDiv.appendChild(messageContent);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function embedYouTubeVideos(link) {
    const videoContainer = document.getElementById("video-container");
    const iframe = document.createElement("iframe");
    iframe.className = "youtube-video";
    iframe.src = link.replace("watch?v=", "embed/");
    iframe.width = "300";  // Adjust width
    iframe.height = "169"; // Adjust height to maintain 16:9 aspect ratio
    iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
    iframe.allowFullscreen = true;
    videoContainer.appendChild(iframe);
}

function clearVideoContainer() {
    const videoContainer = document.getElementById("video-container");
    videoContainer.innerHTML = "";  // Clear existing videos
}
