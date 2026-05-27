$ErrorActionPreference = 'Stop'
$loginBody = @{ username = 'admin'; password = 'Admin123@' } | ConvertTo-Json
$login = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/auth/login' -ContentType 'application/json' -Body $loginBody
$token = $login.token
$headers = @{ Authorization = "Bearer $token" }
$uri = 'http://127.0.0.1:8000/chat'

function Ask-Chat {
    param(
        [string]$Label,
        [string]$Question,
        [string[]]$History = @(),
        [string]$ChatId = $null
    )
    $payload = @{ question = $Question; conversation_history = $History }
    if ($ChatId) { $payload.chat_id = $ChatId }
    $json = $payload | ConvertTo-Json -Depth 8
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Method Post -Uri $uri -Headers $headers -ContentType 'application/json' -Body $json -TimeoutSec 300
    $sw.Stop()
    [PSCustomObject]@{
        label = $Label
        question = $Question
        elapsed_seconds = [math]::Round($sw.Elapsed.TotalSeconds,2)
        chat_id = $resp.chat_id
        answer_mode = $resp.answer_mode
        confidence_score = if ($resp.confidence.score -ne $null) { $resp.confidence.score } else { $null }
        confidence_level = if ($resp.confidence.level -ne $null) { $resp.confidence.level } else { $null }
        branch = if ($resp.classification.main_branch -ne $null) { $resp.classification.main_branch } else { $null }
        follow_up = if ($resp.classification.is_follow_up -ne $null) { $resp.classification.is_follow_up } else { $null }
        validation_issues = @($resp.validation_issues | ForEach-Object { $_.code })
        sources = @($resp.sources | ForEach-Object { if ($_.article_number) { "$($_.title) [art. $($_.article_number)]" } else { $_.title } } | Select-Object -Unique)
        answer = $resp.answer
    }
}

$results = @()

$r1 = Ask-Chat -Label 'civil-imovel' -Question 'Um contrato de compra e venda de imóvel pode ser resolvido por incumprimento? Indique fundamento legal e diga se o contexto recuperado confirma também os prazos.'
$results += $r1

$r2 = Ask-Chat -Label 'laboral-despedimento' -Question 'Fui despedido sem justa causa. Tenho direito a indemnização e reintegração? Responda em linguagem simples e diga se o prazo para contestar está confirmado no material recuperado.'
$results += $r2
$hist2 = @("Utilizador: $($r2.question)", "Assistente: $($r2.answer)")
$r3 = Ask-Chat -Label 'laboral-followup' -Question 'E se eu não quiser voltar para a empresa e só quiser dinheiro, muda alguma coisa?' -History $hist2 -ChatId $r2.chat_id
$results += $r3

$r4 = Ask-Chat -Label 'penal-burla' -Question 'Em que circunstâncias se configura a burla qualificada e qual é a pena aplicável? Se o corpus não confirmar, diga isso claramente.'
$results += $r4

$r5 = Ask-Chat -Label 'sucessoes-menores' -Question 'Como se processa a partilha de herança quando existem herdeiros menores em Angola? Qual é a intervenção do tribunal e dos pais ou encarregados?'
$results += $r5

$r6 = Ask-Chat -Label 'bi-custo' -Question 'Qual é o custo da segunda via do Bilhete de Identidade e onde normalmente se trata isso?'
$results += $r6

$r7 = Ask-Chat -Label 'artigo-exacto' -Question 'O que diz o artigo 26 da Constituição angolana?'
$results += $r7
$hist7 = @("Utilizador: $($r7.question)", "Assistente: $($r7.answer)")
$r8 = Ask-Chat -Label 'artigo-followup' -Question 'Esse mesmo artigo também se aplica contra particulares?' -History $hist7 -ChatId $r7.chat_id
$results += $r8

$results | ConvertTo-Json -Depth 8
