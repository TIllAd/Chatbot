package de.fau.wisochatbot.chat;
import java.util.Map;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

// src/main/java/.../ChatController.java
@RestController
@RequestMapping("/api")
public class ChatController {

  private final LlmService llmService;

  public ChatController(LlmService llmService) {
    this.llmService = llmService;
  }

  public record ChatRequest(String message) {}

  public record ChatResponse(String reply) {}

  @PostMapping("/chat")
  public Map<?, ?> chat(
      @RequestBody ChatRequest req,
      @RequestParam(defaultValue = "false") boolean debug) {
    return llmService.reply(req.message(), debug);
  }
}
