package de.fau.wisochatbot.api;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chat")
public class ChatApiController {

    @PostMapping
    public ChatResponse chat(@RequestBody ChatRequest request) {

        String userMessage = request.message();

        String reply = "Du: " + userMessage +
                "\nBot: (hier kommt später RAG/LLM rein)";

        return new ChatResponse(reply);
    }
}