package de.fau.wisochatbot.chat;

import jakarta.servlet.http.HttpSession;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class ChatController {

    private final LlmService llmService;

    public ChatController(LlmService llmService) {
        this.llmService = llmService;
    }

    public record ChatRequest(String message) {
    }

    public record ChatResponse(String reply) {
    }

    @PostMapping("/chat")
    public ChatResponse chat(@RequestBody ChatRequest req, HttpSession session) {

        @SuppressWarnings("unchecked")
        List<Map<String, String>> history =
                (List<Map<String, String>>) session.getAttribute("history");

        if (history == null) {
            history = new ArrayList<>();
            history.add(Map.of(
                    "role", "system",
                    "content", "Du bist der WiSo-Chatbot. Antworte kurz und hilfreich auf Deutsch."
            ));
        }

        // User Nachricht hinzufügen
        history.add(Map.of("role", "user", "content", req.message()));

        // LLM mit kompletter History aufrufen
        String answer = llmService.replyWithHistory(history);

        // Antwort zur History hinzufügen
        history.add(Map.of("role", "assistant", "content", answer));

        int maxHistorySize = 21; // 1 System + 20 letzte Nachrichten

        if (history.size() > maxHistorySize) {
            Map<String, String> system = history.get(0);

            List<Map<String, String>> lastMessages = history.subList(history.size() - 20, history.size());

            history = new ArrayList<>();
            history.add(system);
            history.addAll(lastMessages);
        }

        session.setAttribute("history", history);

        return new ChatResponse(answer);
    }

    @PostMapping("/chat/reset")
    public void reset(HttpSession session) {
        session.removeAttribute("history");
    }
}