$ErrorActionPreference = 'Stop'
$token = '6effe929644ce42094848d8688641008924f98e7b1b03c38c3a5080c49bb6deb'
$headers = @{ Authorization = "Bearer $token" }
$uri = 'http://127.0.0.1:8000/chat'

function Ask-Chat {
    param(
        [string]$Question,
        [string[]]$History = @(),
        [string]$ChatId = $null,
        [string]$Label
    )

    $payload = @{
        question = $Question
        conversation_history = $History
    }
    if ($ChatId) { $payload.chat_id = $ChatId }

    $json = $payload | ConvertTo-Json -Depth 8
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Method Post -Uri $uri -Headers $headers -ContentType 'application/json' -Body $json -TimeoutSec 300
    $sw.Stop()

    [PSCustomObject]@{
        label = $Label
        question = $Question
        elapsed_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        chat_id = $resp.chat_id
        answer_mode = $resp.answer_mode
        provider_used = $resp.provider_used
        confidence_score = if ($resp.confidence.score -ne $null) { $resp.confidence.score } else { $null }
        confidence_label = if ($resp.confidence.label -ne $null) { $resp.confidence.label } else { $null }
        classification = $resp.classification
        clarifying_questions = $resp.clarifying_questions
        validation_issues = $resp.validation_issues
        source_titles = @($resp.sources | ForEach-Object { $_.title } | Select-Object -Unique)
        source_articles = @($resp.sources | ForEach-Object { $_.article_number } | Where-Object { $_ })
        answer_preview = if ($resp.answer.Length -gt 1200) { $resp.answer.Substring(0, 1200) } else { $resp.answer }
        full_answer = $resp.answer
    }
}

$results = @()

$r1 = Ask-Chat -Label 'laboral-1' -Question 'O meu chefe deixou de me pagar o salário há dois meses. O que posso exigir e quais são os próximos passos em Angola?'
$results += $r1
$hist1 = @(
    "Utilizador: $($r1.question)",
    "Assistente: $($r1.full_answer)"
)
$r2 = Ask-Chat -Label 'laboral-2-followup' -Question 'E se além disso ele me despedir sem aviso prévio, muda alguma coisa?' -History $hist1 -ChatId $r1.chat_id
$results += $r2

$r3 = Ask-Chat -Label 'bi-1' -Question 'Quanto custa a segunda via do Bilhete de Identidade e onde normalmente se trata isso?'
$results += $r3

$r4 = Ask-Chat -Label 'ambiguo-1' -Question 'Preciso de ajuda com a lei, o que faço agora?'
$results += $r4

$r5 = Ask-Chat -Label 'constitucional-1' -Question 'O que diz o artigo 26 da Constituição angolana e qual é a ideia principal dele?'
$results += $r5
$hist5 = @(
    "Utilizador: $($r5.question)",
    "Assistente: $($r5.full_answer)"
)
$r6 = Ask-Chat -Label 'constitucional-2-followup' -Question 'Esse artigo também pode ser usado contra actos de particulares ou é mais contra o Estado?' -History $hist5 -ChatId $r5.chat_id
$results += $r6

$r7 = Ask-Chat -Label 'sociedades-1' -Question 'Numa sociedade por quotas, que direitos práticos têm os sócios minoritários para se defenderem da maioria?'
$results += $r7
$hist7 = @(
    "Utilizador: $($r7.question)",
    "Assistente: $($r7.full_answer)"
)
$r8 = Ask-Chat -Label 'sociedades-2-followup' -Question 'E eles podem forçar convocação de assembleia ou pedir informações da empresa?' -History $hist7 -ChatId $r7.chat_id
$results += $r8

$r9 = Ask-Chat -Label 'processual-1' -Question 'Qual é o prazo de recurso no CPP para prisão preventiva?'
$results += $r9

$r10 = Ask-Chat -Label 'arrendamento-1' -Question 'Um senhorio pode despejar-me sem justo motivo e sem decisão judicial?'
$results += $r10

$results | ConvertTo-Json -Depth 8
