document
  .getElementById("message_area")
  .addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent the default form submission

    const chatInput = document.getElementById("text");
    const chatHistory = document.getElementById("chatHistory");
    const userMessage = chatInput.value.trim();

    if (userMessage === "") {
      return;
    }

    // Display the user's message in the chat history
    const userMessageDiv = document.createElement("div");
    userMessageDiv.classList.add("response_tab_user", "user");
    userMessageDiv.textContent = userMessage;
    chatHistory.appendChild(userMessageDiv);

    // Clear the input field
    chatInput.value = "";

    try {
      // Send the user's message to the FastAPI backend
      const response = await fetch('http://127.0.0.1:8000/execute_query/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query_sentence: userMessage }), // Correct field name
      });

      if (response.ok) {
        const data = await response.json();
        const sqlQuery = data.query;
        const results = data.results;

        // Display the generated SQL query in the chat history
        const sqlQueryDiv = document.createElement("div");
        sqlQueryDiv.classList.add("response_tab_assistant", "assistant");
        sqlQueryDiv.textContent = "Generated SQL Query: " + sqlQuery;
        chatHistory.appendChild(sqlQueryDiv);

        // Display the SQL query results in the chat history
        if (results.length > 0) {
          const resultsDiv = document.createElement("div");
          resultsDiv.classList.add("response_tab_assistant", "assistant");

          const table = document.createElement("table");
          const thead = document.createElement("thead");
          const tbody = document.createElement("tbody");

          // Set table headers
          const headers = Object.keys(results[0]);
          const headerRow = document.createElement("tr");
          headers.forEach((header) => {
            const th = document.createElement("th");
            th.innerText = header;
            headerRow.appendChild(th);
          });
          thead.appendChild(headerRow);

          // Set table rows
          results.forEach((row) => {
            const tr = document.createElement("tr");
            headers.forEach((header) => {
              const td = document.createElement("td");
              td.innerText = row[header];
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          });

          table.appendChild(thead);
          table.appendChild(tbody);
          resultsDiv.appendChild(table);
          chatHistory.appendChild(resultsDiv);
        } else {
          const noResultsDiv = document.createElement("div");
          noResultsDiv.classList.add("response_tab_assistant", "assistant");
          noResultsDiv.textContent = "No results found.";
          chatHistory.appendChild(noResultsDiv);
        }

        // Scroll to the bottom of the chat history
        chatHistory.scrollTop = chatHistory.scrollHeight;
      } else {
        console.error("Failed to send message to the backend");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  });
